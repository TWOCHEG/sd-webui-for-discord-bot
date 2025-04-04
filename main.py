import disnake
from disnake.ext import commands
from disnake import Locale, Localized
import asyncio
import os
import aiohttp
from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageChops
import base64
from io import BytesIO
from langdetect import detect  # говнище
import random
import pynvml

import torch
from transformers import CLIPProcessor, CLIPModel

from deep_translator import GoogleTranslator

model = None
processor = None


class ImageGenLocal(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.queue = {}

        self.usage = {}
        self.max_usage = 3

        self.current_class: System = None

        self.interrupt_id = 'generation_interrupt'

    @commands.cooldown(1, 60, commands.BucketType.user)
    @commands.slash_command(
        name='automatic1111',
        description='AUTOMATIC1111/stable-diffusion-webui'
    )
    async def generate_a1111(
        self,
        inter,
        checkpoint: str = commands.Param(
            name=Localized(
                'model',
                data={Locale.ru: 'модель'}
            ),
            choices=[
                disnake.OptionChoice(name, name.replace('.' + name.split('.')[-1], '')) for name in
                os.listdir(os.environ['SDWEBUI'] + '/webui/models/Stable-diffusion')
            ],
            default=os.listdir(os.environ['SDWEBUI'] + '/webui/models/Stable-diffusion')[0]
        ),
        lora: str = commands.Param(
            name=Localized(
                'lora',
                data={Locale.ru: 'лора'}
            ),
            description=Localized(
                'Low Adaption Rank (i\'ve helped a lot 😀)',
                data={Locale.ru: 'мне лень, узнай сам что это'}
            ),
            choices=[
                disnake.OptionChoice(name, name.replace('.' + name.split('.')[-1], '')) for name in
                os.listdir(os.environ['SDWEBUI'] + '/webui/models/Lora')
            ],
            default=None
        ),
        prompt: str = commands.Param(
            name=Localized(
                'prompt',
                data={Locale.ru: 'запрос'}
            ),
        ),
        negative_prompt: str = commands.Param(
            name=Localized(
                'negative_prompt',
                data={Locale.ru: 'отрицательная_подсказка'}
            ),
            default=''
        ),
        batch_size: int = commands.Param(
            name=Localized(
                'batch_size',
                data={Locale.ru: 'размер_партии'}
            ),
            default=1
        ),
        size: str = commands.Param(
            name=Localized(
                'size',
                data={Locale.ru: 'размеры'}
            ),
            choices=[
                disnake.OptionChoice(
                    Localized(
                        'default (1024 x 1024)',
                        data={Locale.ru: 'стандарт (1024 x 1024)'}
                    ),
                    '512 512'
                ),
                disnake.OptionChoice(
                    Localized(
                        'height (2048 x 1024)',
                        data={Locale.ru: 'высота (2048 x 1024)'}
                    ),
                    '1024 512'
                ),
                disnake.OptionChoice(
                    Localized(
                        'width (1024 x 2048)',
                        data={Locale.ru: 'ширина (1024 x 2048)'}
                    ),
                    '512 1024'
                ),
            ],
            default='512 512'
        ),
        auto_translate: int = commands.Param(
            name=Localized(
                'auto_translate',
                data={Locale.ru: 'автоматический_перевод'}
            ),
            choices=[
                disnake.OptionChoice('True', 1),
                disnake.OptionChoice('False', 0),
            ],
            default=1
        ),
    ):
        async def process(checkpoint, prompt, negative_prompt, batch_size, height, width, auto_translate, lora):
            locale = inter.locale == Locale.ru
            user_id = inter.user.id

            if self.usage.get(user_id, 0) > self.max_usage:
                await inter.send(
                    embed=disnake.Embed(
                        title='ERR',
                        description=(
                            f'ты превысил свои лимиты в сутки `{self.max_usage}`'
                            if locale else 
                            f'you exceeded your daily limits `{self.max_usage}`'
                        )
                    ),
                    ephemeral=True
                )
                return
            self.usage[user_id] = self.usage.setdefault(user_id, 0) + 1
            if user_id in self.queue:
                await inter.send(
                    embed=disnake.Embed(
                        title='ERR',
                        description=(
                            'твое изображение еще делается' if locale else 'your image is still being made'
                        ) + f'\n```{prompt}```\n' + (f'```{negative_prompt}```' if negative_prompt else '')
                    ),
                    ephemeral=True
                )
                return

            await inter.send('начинаю...' if locale else 'starting...')
            message = await inter.original_response()
            self.queue[user_id] = len(self.queue) + 1
            if len(self.queue) >= 1:
                await message.edit(f'очередь... `{self.queue[user_id]}`' if locale else f'queue... `{self.queue[user_id]}`')
                while self.queue[user_id] > 1:
                    await asyncio.sleep(1)
            await message.edit('генерация...' if locale else 'generating...')

            request_params = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'override_settings': {
                    'sd_model_checkpoint': checkpoint
                },
                'width': max(min(width, 1024), 512),
                'height': max(min(height, 1024), 512),
                'batch_size': max(min(batch_size, 4), 1)
            }
            if lora:
                request_params['override_settings']['sd_lora'] = lora

            self.current_class = System(
                bot=self.bot,
                generation_interrupt=self.interrupt_id,
                request_params=request_params
            )
            await asyncio.create_task(
                self.current_class.main(inter, auto_translate)
            )
            await self.current_class.move_queue(inter.author.id)

            if self.usage.get(user_id, 0) > self.max_usage:
                await asyncio.sleep(86400)

                if user_id in self.usage:
                    del self.usage[user_id]

        await asyncio.create_task(
            process(
                checkpoint=checkpoint, 
                prompt=prompt, 
                negative_prompt=negative_prompt,
                batch_size=batch_size,
                height=int(size.split()[0]),
                width=int(size.split()[1]),
                auto_translate=bool(auto_translate),
                lora=lora,
            )
        )

    @commands.Cog.listener()
    async def on_button_click(self, inter):
        if inter.message.author != self.bot.user:
            return
        
        locale = inter.locale == Locale.ru
        custom_id: str = inter.data.custom_id

        if custom_id.startswith(self.interrupt_id):
            await inter.response.defer(ephemeral=True)
            if not int(custom_id.split('_')[-1]) == inter.author.id:
                await inter.send(
                    'нет'
                    if locale else
                    'no',
                    ephemeral=True
                )
                return
            if not self.current_class:
                await inter.send(
                    'класс не найден'
                    if locale else
                    'class not found',
                    ephemeral=True
                )
                return

            await self.current_class.interrupt(inter)

class System(ImageGenLocal):
    def __init__(
        self,
        bot,
        generation_interrupt: str,
        request_params: dict = {
            'prompt': '',
            'negative_prompt': '',
            'override_settings': {
                'sd_model_checkpoint': '.safetensor'
            },
            'width': 512,
            'height': 512,
            'batch_size': 1
        },
        img_check_attempts: int = 1,
        sleep_time: int = 6,
        gpu_overload: int = 95
    ):
        super().__init__(bot)
        
        self.results = None

        self.generation_interrupt = generation_interrupt
        # константы
        self.img_check_attempts = img_check_attempts
        self.sleep_time = sleep_time
        self.gpu_overload = gpu_overload
        # запросы
        self.request_params = request_params
        self.payload = {
            "steps": 40,
            "cfg_scale": 7.5,
            "enable_hr": True,
            "hr_scale": 2,
            "hr_upscaler": "Latent",
            "hr_steps": 15,
            "denoising_strength": 0.7,
        }
        # счетчики попыток
        self.img_check_count = 0
        # если изоражение NSFW
        self.ephemeral = False
        self.nsfw_channel = False
        self.warn_image_create = False

    async def gen_request(self):
        url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

        payload = self.payload | self.request_params
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.results = [Image.open(BytesIO(base64.b64decode(img))) for img in result["images"]]
                    else:
                        self.results = f'`{response.status}`\n{await response.text()}'
            except Exception as e:
                self.results = str(e)

    async def main(self, inter: disnake.AppCommandInteraction, auto_translate: bool = True):
        locale = inter.locale == Locale.ru

        task = asyncio.create_task(self.gen_request())

        keys = self.request_params
        
        prompt, negative_prompt = keys.get('prompt', ''), keys.get('negative_prompt', '')
        if auto_translate:
            prompt_translit, negative_prompt_translit = await self.get_translit(prompt, negative_prompt)
        else:
            prompt_translit, negative_prompt_translit = None, None
        keys['prompt'] = prompt_translit or prompt
        keys['negative_prompt'] = negative_prompt_translit or negative_prompt

        prompts_embed = disnake.Embed(
            title=(
                'запрос'
            ) if locale else (
                'request'
            ),
            description=(
                f'промпт: `{keys['prompt']}`\n'
                f'отрицательная подзказка: `{keys['negative_prompt'] or 'none'}`\n'
                f'модель: `{keys['override_settings']['sd_model_checkpoint']}`\n'
                f'ширина, высота: `{keys.get('width', 0) * 2}`, `{keys.get('height', 0) * 2}`\n'
                f'кол-во изображений: `{keys.get('batch_size')}`'
            ) if locale else (
                f'prompt: `{keys['prompt']}`\n'
                f'negative_prompt: `{keys['negative_prompt'] or 'none'}`\n'
                f'checkpoint: `{keys['override_settings']['sd_model_checkpoint']}`\n'
                f'width, height: `{keys.get('width', 0) * 2}`, `{keys.get('height', 0) * 2}`\n'
                f'batch size: `{keys.get('batch_size')}`'
            )
        )

        message = await inter.original_response()

        while not self.results:
            if await self.gpu_overload_check(self.gpu_overload):
                task.cancel()
                await self.interrupt()
                await inter.send(
                    'видео память перегружена, процесс остановлен\nпрости меня 🙁'
                    if locale else
                    'the GPU memory is overloaded, the process is stopped\nim sorry 🙁',
                    ephemeral=True
                )
                return

            progress = await self.get_progress()
            
            image = await self.get_image(message, progress.get('current_image'), locale)

            progress_embed = await self.get_embeds(progress, locale)
            
            args = {
                'content': '',
                'embeds': [progress_embed, prompts_embed],
                'components': [
                    disnake.ui.Button(
                        label='прервать' if locale else 'interrupt',
                        style=disnake.ButtonStyle.red,
                        custom_id=self.generation_interrupt + f'_{inter.author.id}'
                    )
                ]
            }
            if image:
                args['files'] = await self.get_files(image)
                args['attachments'] = []

            await message.edit(
                **args
            )

            await asyncio.sleep(self.sleep_time)

        await self.send_result(
            inter,
            message,
            prompt, 
            negative_prompt,
            prompt_translit,
            negative_prompt_translit
        )
        
    async def send_result(
        self,
        inter: disnake.AppCommandInter, 
        message,
        prompt, 
        negative_prompt, 
        prompt_translit, 
        negative_prompt_translit
    ):
        locale = inter.locale == Locale.ru
        if isinstance(self.results, str):
            await inter.send(
                embed=disnake.Embed(
                    title=(
                        'ERR'
                    ) if locale else (
                        'ОШИБКА'
                    ),
                    description=self.results
                ),
                ephemeral=True
            )
            return
        args = {
            'content': (
                f"- **промпт**\n> {prompt}\n"
                f"{f'- перевод\n> {prompt_translit}\n' if prompt_translit else ''}"
                f"- **отрицательная подсказка**\n> {negative_prompt or 'нет'}\n"
                f"{f'- перевод\n> {negative_prompt_translit}\n' if negative_prompt_translit else ''}"
            ) if locale else (
                f"- **prompt**\n> {prompt}\n"
                f"{f'- translit\n> {prompt_translit}\n' if prompt_translit else ''}"
                f"- **negative prompt**\n> {negative_prompt or 'none'}\n"
                f"{f'- translit\n> {negative_prompt_translit}\n' if negative_prompt_translit else ''}"
            ) + f'-# {self.request_params['override_settings']['sd_model_checkpoint']}',
            'files': await self.get_files(self.results),
            'embeds': [],
            'components': []
        }
        if self.ephemeral:
            args['ephemeral'] = True
        else:
            args['attachments'] = []
        if self.ephemeral:
            await inter.send(
                **args
            )
        else:
            try:
                await message.edit(
                    **args
                )
            except disnake.NotFound:
                await inter.send(
                    **args
                )

    async def get_files(self, images: list):
        if not isinstance(images, list):
            images = [images]
        files = []
        for i, image in enumerate(images):
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='png')
            img_byte_arr.seek(0)

            files.append(
                disnake.File(fp=img_byte_arr, filename=f'image_{i}.png')
            )

        return files

    async def get_translit(self, prompt, negative_prompt):
        prompt_translit = None
        negative_prompt_translit = None
        try:
            prompt_translit = self.translate_text(prompt) if detect(prompt) != 'en' else None
        except:
            pass
        try:
            negative_prompt_translit = self.translate_text(negative_prompt) if detect(negative_prompt) != 'en' else None
        except:
            pass
        return prompt_translit, negative_prompt_translit

    async def move_queue(self, user_id):
        del self.queue[user_id]
        self.queue = {user_id: pos - 1 for user_id, pos in self.main_class.queue.items() if pos > 1}

    async def get_progress(self):
        url = 'http://127.0.0.1:7860/sdapi/v1/progress'
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'current_image' in result:
                            try:
                                result['current_image'] = Image.open(
                                    BytesIO(base64.b64decode(result['current_image']))
                                )
                            except:
                                del result['current_image']
                        return result
                    else:
                        return {}
        except:
            return {}

    async def get_embeds(self, progress, locale):
        if progress:
            text = {
                True: {
                    'checking': f'попытка проверки: `{self.img_check_count} / {self.img_check_attempts}`\n',
                    True: 'проверка завершена: `распознан nsfw`\n',
                    'nsfw_channel': 'проверка завершена: `канал nsfw, проверка отклонена`\n',
                    False: 'проверка завершена: `nsfw не распознан`\n'
                },
                False: {
                    'checking': f'check attempt: `{self.img_check_count} / {self.img_check_attempts}`\n',
                    True: 'verify completed: `detected nsfw`\n',
                    'nsfw_channel': 'verify completed: `nsfw channel, verification rejected`\n',
                    False: 'verify completed: `nsfw is not detected`\n'
                }
            }[locale]
            if self.img_check_count < self.img_check_attempts and not (self.ephemeral or self.nsfw_channel):
                text = text['checking']
            elif self.ephemeral:
                text = text[True]
            elif self.nsfw_channel:
                text = text['nsfw_channel']
            else:
                text = text[False]
            try:
                total_progress = (progress.get('state', {}).get('sampling_step', 1) / progress.get('state', {}).get('sampling_steps', 1)) * 100
            except:
                total_progress = 'none'

            progress_embed = disnake.Embed(
                title='прогресс' if locale else 'progress',
                description=(
                    f'**вывод**\n'
                    f'общий прогресс: `{progress.get('progress', 0) * 100}%`\n'
                    f'{text}'
                    f'\n**текущее изображение**\n'
                    f'прогресс: `{total_progress}%`\n'
                    f'шаг генерации: `{progress.get('state', {}).get('sampling_step')}`'
                ) if locale else (
                    f'**conclusion**\n'
                    f'total progress: `{progress.get('progress', 0) * 100}%`\n'
                    f'{text}'
                    f'\n**current image**\n'
                    f'progress: `{total_progress}%`\n'
                    f'gen step: `{progress.get('state', {}).get('sampling_step')}`'
                )
            )
        else:
            progress_embed = disnake.Embed(
                title='прогресс' if locale else 'progress',
                description=(
                    'не удалось получить прогресс'
                ) if locale else (
                    'couldn\'t get progress'
                )
            )
        return progress_embed

    async def get_image(self, message, image, locale):
        if image and not self.warn_image_create:
            if message.channel.is_nsfw():
                self.nsfw_channel = True
            if self.img_check_attempts > self.img_check_count and not (self.ephemeral or self.nsfw_channel):
                self.img_check_count += 1
                check_result = await self.check_image(image)
                if check_result:
                    max_key = max(check_result, key=check_result.get).lower()
                    if max_key not in ['violent', 'shocking']:
                        self.ephemeral = True

            if self.ephemeral:
                if image:
                    image = await self.add_warn_to_img(
                        image,
                        (
                            ['обнаружено порно', 'просьба сохранять спокойствие']
                            if locale else 
                            ['NSFW detected', 'please remain calm']
                        ),
                        Image.open(random.choice(
                            ['assets/img/' + file for file in os.listdir('assets/img/') if file.startswith('warning_image')]
                        ))
                    )
                    self.warn_image_create = True
        else:
            image = None
        return image

    async def interrupt(self, inter: disnake.AppCommandInter = None):
        async with aiohttp.ClientSession() as session:
            if inter:
                locale = inter.locale == Locale.ru
                try:
                    async with session.post('http://127.0.0.1:7860/sdapi/v1/interrupt') as response:
                        if response.status == 200:
                            await inter.send(
                                'останавливаю...'
                                if locale else 
                                'stopping...',
                                ephemeral=True
                            )
                        else:
                            text = await response.text()
                            await inter.send(
                                f'ошибка: {text}'
                                if locale else 
                                f'error: {text}',
                                ephemeral=True
                            )
                except Exception as e:
                    await inter.send(
                        f'ошибка: {e}'
                        if locale else 
                        f'error: {e}',
                        ephemeral=True
                    )
    
    async def gpu_overload_check(self, threshold: int = 90):
        result = False
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                total_memory = mem_info.total
                used_memory = mem_info.used
                
                usage_percent = (used_memory / total_memory) * 100
                
                if usage_percent > threshold:
                    result = True
                
        except pynvml.NVMLError:
            pass
        finally:
            pynvml.nvmlShutdown()

        return result

    async def add_warn_to_img(self, img: Image.Image, text_list: list[str], icon: Image.Image) -> Image.Image:
        size = img.size
        width, height = size
        aspect_ratio = width / height
        img = img.convert("RGB")
        if aspect_ratio > 1:
            new_width = 2048
            new_height = 1024
        elif aspect_ratio < 1:
            new_width = 1024
            new_height = 2048
        else:
            new_width = 1024
            new_height = 1024
        img = img.resize((new_width, new_height))
        
        blurred = img.filter(ImageFilter.GaussianBlur(radius=50))

        overlay_img = icon.convert('RGBA')
        size = blurred.size
        overlay_img = overlay_img.resize((int(size[0] / 2), int(size[1] / 2)))
        white_bg = Image.new('RGBA', overlay_img.size, (255, 255, 255, 255))
        diff = ImageChops.difference(overlay_img, white_bg)
        bbox = diff.getbbox()
        if bbox:
            overlay_img = overlay_img.crop(bbox)
        bg_width, bg_height = blurred.size
        img_width, img_height = overlay_img.size
        offset = ((bg_width - img_width) // 2, 10)
        
        shadow = Image.new('RGBA', overlay_img.size, color=(0, 0, 0, 255))
        alpha = overlay_img.split()[-1]
        shadow.putalpha(alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
        shadow_offset = (offset[0] + 10, offset[1] + 10)
        blurred.paste(shadow, shadow_offset, mask=shadow)
        blurred.paste(overlay_img, offset, mask=overlay_img)

        text_mask = Image.new("L", blurred.size, 0)
        draw = ImageDraw.Draw(text_mask)
        margin_ratio = 0.05
        max_text_width = blurred.width * (1 - 2 * margin_ratio)
        font_path = "comicbd.ttf"
        initial_font_size = blurred.width // 10
        min_font_size = 10
        line_spacing = 20
        
        def load_font(size):
            try:
                return ImageFont.truetype(font_path, max(size, min_font_size))
            except IOError:
                try:
                    return ImageFont.truetype("arial.ttf", max(size, min_font_size))
                except IOError:
                    return ImageFont.load_default()

        def get_text_size(font_obj, text):
            try:
                bbox = draw.textbbox((0, 0), text, font=font_obj)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                return font_obj.getsize(text)

        fonts = []
        text_sizes = []
        for text in text_list:
            font_size = initial_font_size
            current_font = load_font(font_size)
            text_width, text_height = get_text_size(current_font, text)
            
            if text_width > max_text_width:
                while text_width > max_text_width and font_size > min_font_size:
                    font_size -= 1
                    current_font = load_font(font_size)
                    text_width, text_height = get_text_size(current_font, text)
            else:
                while text_width < max_text_width:
                    font_size += 1
                    current_font = load_font(font_size)
                    new_width, new_height = get_text_size(current_font, text)
                    if new_width > max_text_width:
                        font_size -= 1
                        current_font = load_font(font_size)
                        text_width, text_height = get_text_size(current_font, text)
                        break
                    text_width, text_height = new_width, new_height
            
            fonts.append(current_font)
            text_sizes.append((text_width, text_height))

        total_text_height = sum(height for _, height in text_sizes) + (len(text_list) - 1) * line_spacing
        start_y = ((blurred.height - total_text_height) // 2) + 150
        current_y = start_y

        text_layer = Image.new("RGBA", blurred.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        for i, (text, font) in enumerate(zip(text_list, fonts)):
            text_width, text_height = text_sizes[i]
            x = (blurred.width - text_width) // 2
            
            shadow_color = (0, 0, 0, 255)
            shadow_offset = (x + 10, current_y + 10)
            text_draw.text(shadow_offset, text, font=font, fill=shadow_color)
            
            draw.text((x, current_y), text, fill=255, font=font)
            
            current_y += text_height + line_spacing

        text_shadow = text_layer.filter(ImageFilter.GaussianBlur(radius=10))
        blurred.paste(text_shadow, (0, 0), mask=text_shadow)

        inverted = ImageChops.invert(blurred)
        result = Image.composite(inverted, blurred, text_mask)
        return result

    async def translate_text(text, dest_language='en'):
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated

    async def check_image(self, image: Image.Image, thresholds: dict = {'neutral': 0.4, 'porno': 0.6, 'violent': 0.6, 'shocking': 0.6}) -> dict | bool:
        global model, processor

        try:
            if model is None:
                model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
            if processor is None:
                processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
        except:
            return

        image = image.convert('RGB')
        labels = list(thresholds.keys())

        inputs = processor(text=labels, images=image, return_tensors='pt', padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        result = {label: prob.item() for label, prob in zip(labels, probs)}
        for label, prob in result.items():
            if prob > thresholds[label]:
                if label in ['safe', 'neutral']:
                    return
                return result

        return
