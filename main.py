import disnake
from disnake.ext import commands
from disnake import Locale, Localized
import asyncio
import os
import aiohttp
from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageChops
import base64
from io import BytesIO
from langdetect import detect
import random
import torch
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator


class System:
    def __init__(
        self, 
        main_class,
        request_params: dict = {
            'prompt': '',
            'negative_prompt': '',
            'checkpoint': '',
            'width': 512,
            'height': 512,
            'batch_size': 1
        },
        img_check_attempts: int = 3, 
        img_create_freq: int = 6
    ):
        self.results = None

        self.main_class = main_class
        # константы попыток
        self.img_check_attempts = img_check_attempts
        self.img_create_freq = img_create_freq
        # запросы
        self.request_params = request_params
        self.payload = {
            "steps": 25,
            "cfg_scale": 7.5,
            "enable_hr": True,
            "hr_scale": 2,
            "hr_upscaler": "Latent",
            "hr_steps": 15,
            "denoising_strength": 0.7,
        }
        # счетчики попыток
        self.img_check_count = 0
        self.img_create_count = 0
        # если изоражение NSFW
        self.ephemeral = False
        self.nsfw_channel = False

    async def gen_request(self):
        url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

        payload = self.payload | self.request_params
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        result_to_print = result.copy()
                        if 'images' in result_to_print:
                            result_to_print['images'] = type(result_to_print['images'])
                        self.results = [Image.open(BytesIO(base64.b64decode(img))) for img in result["images"]]
                    else:
                        self.results = f'`{response.status}`\n{await response.text()}'
            except asyncio.TimeoutError as e:
                self.results = str(e)

    async def main(self, inter):
        locale = inter.locale == Locale.ru

        asyncio.create_task(self.gen_request())

        keys = self.request_params

        prompt, negative_prompt = keys.get('prompt', ''), keys.get('negative_prompt', '')
        prompt_translit, negative_prompt_translit = await self.get_translit(prompt, negative_prompt)
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
                f'модель: `{keys['checkpoint']}`\n'
                f'ширина, высота: `{keys['width']}`, `{keys['height']}`\n'
                f'кол-во изображений: `{keys['batch_size']}`'
            ) if locale else (
                f'prompt: `{keys['prompt']}`\n'
                f'negative_prompt: `{keys['negative_prompt'] or 'none'}`\n'
                f'checkpoint: `{keys['checkpoint']}`\n'
                f'width, height: `{keys['width']}`, `{keys['height']}`\n'
                f'batch size: `{keys['batch_size']}`'
            )
        )

        message = await inter.original_response()

        while not self.results:
            await asyncio.sleep(5)

            progress = await self.get_progress()
            if not progress:
                await message.edit(
                    (
                        'не удалось получить прогресс'
                        if locale else 
                        'couldn\'t get progress'
                    ),
                    embed=prompts_embed
                )
                continue

            image = None
            if progress.get('current_image'):
                image = progress['current_image']

                if self.img_check_attempts > self.img_check_count and not self.nsfw_channel:
                    self.img_check_count += 1
                    check_result = await self.check_image(image)
                    if check_result:
                        try:
                            if message.channel.is_nsfw():
                                max_key = max(check_result, key=check_result.get).lower()
                                if max_key not in ['violent', 'shocking']:
                                    self.nsfw_channel = True
                                    self.ephemeral = False
                        except:
                            pass
                        if not self.nsfw_channel:
                            self.ephemeral = True

            if self.ephemeral and image:
                if self.img_create_count > self.img_create_freq:
                    self.img_create_count = 0
                    
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
                else:
                    self.img_create_count += 1
            
            embeds = [
                disnake.Embed(
                    title='прогресс' if locale else 'progress',
                    description=(
                        f'общий прогресс: `{progress.get('progress', 0) * 100}%`\n'
                        f'{'изображение не найдено\n' if not image else ''}'
                        f'\n**текущее изображение**\n'
                        f'прогресс: `{progress.get('sampling_step', 1) / progress.get('sampling_steps', 1) * 100}%`\n'
                        f'шаг генерации: `{progress.get('sampling_step')}`'
                    ) if locale else (
                        f'total progress: `{progress.get('progress', 0) * 100}%`\n'
                        f'{'image not found\n' if not image else ''}'
                        f'\n**current image**\n'
                        f'progress: `{progress.get('sampling_step', 1) / progress.get('sampling_steps', 1) * 100}%`\n'
                        f'gen step: `{progress.get('sampling_step')}`'
                    )
                ),
                prompts_embed
            ]

            if image:
                files = await self.get_files(image)
            else:
                files = []

            await message.edit(
                embeds=embeds,
                files=files
            )

        await self.move_queue(inter.author.id)
        await self.send_result(
            inter,
            message,
            prompt, prompt_translit,
            negative_prompt, negative_prompt_translit
        )
        
    async def send_result(
        self,
        inter: disnake.AppCommandInter, 
        message,
        prompt, negative_prompt, 
        prompt_translit, negative_prompt_translit
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
            ),
            'files': await self.get_files(self.results)
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
        del self.main_class.queue[user_id]
        self.main_class.queue = {user_id: pos - 1 for user_id, pos in self.main_class.queue.items() if pos > 1}

    async def get_progress(self, timeout: int = 2):
        url = 'http://127.0.0.1:7860/sdapi/v1/progress'
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
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
                        return None
        except asyncio.TimeoutError:
            return None

    async def add_warn_to_img(self, img: Image.Image, text_list: list[str], icon: Image.Image) -> Image.Image:
        img = img.convert("RGB").resize((1024, 1024))
        
        blurred = img.filter(ImageFilter.GaussianBlur(radius=30))

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
        offset = ((bg_width - img_width) // 2, ((bg_height - img_height) // 2) - 150)
        
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

    async def check_image(image: Image.Image, thresholds: dict = {'safe': 0.5, 'neutral': 0.5, 'nsfw': 0.5, 'violent': 0.7, 'shocking': 0.6}) -> dict | bool:
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

        # Формирование результата
        result = {label: prob.item() for label, prob in zip(labels, probs)}
        for label, prob in result.items():
            if prob > thresholds[label]:
                if label in ['safe', 'neutral']:
                    return
                return result

        return

    def translate_text(text, dest_language='en'):
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated

class ImageGenLocal(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.queue = {}

    @commands.cooldown(1, 60, commands.BucketType.user)
    @commands.slash_command(
        name=Localized(
            'gen_img',
            data={
                Locale.ru: 'сгенерировать_изображение'
            }
        ),
        description=Localized(
            'generate an image',
            data={
                Locale.ru: 'негры негры негры',
            }
        ),
    )
    async def generate(
        self,
        inter,
        checkpoint: str = commands.Param(
            name=Localized(
                'model',
                data={Locale.ru: 'модель'}
            ),
            choices=[
                disnake.OptionChoice(name, name) for name in
                os.listdir(os.environ['SDWEBUI'] + '/webui/models/Stable-diffusion')
            ],
            default=os.listdir(os.environ['SDWEBUI'] + '/webui/models/Stable-diffusion')[4]
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
        height: int = commands.Param(
            name=Localized(
                'height',
                data={Locale.ru: 'высота'}
            ),
            description=Localized(
                '1024 - 512 (the size will be doubled via "Hires. fix")',
                data={Locale.ru: '1024 - 512 (размер будет удвоен через "Hires. fix")'}
            ),
            default=512
        ),
        width: int = commands.Param(
            name=Localized(
                'width',
                data={Locale.ru: 'ширина'}
            ),
            description=Localized(
                '1024 - 512 (the size will be doubled via "Hires. fix")',
                data={Locale.ru: '1024 - 512 (размер будет удвоен через "Hires. fix")'}
            ),
            default=512
        )
    ):
        async def process(checkpoint, prompt, negative_prompt, batch_size, height, width):
            locale = inter.locale == Locale.ru
            user_id = inter.user.id

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

            gen_class = System(
                self,
                {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'checkpoint': checkpoint,
                    'width': max(min(width, 1024), 512),
                    'height': max(min(height, 1024), 512),
                    'batch_size': max(min(batch_size, 4), 1)
                }
            )

            await asyncio.create_task(gen_class.main(inter))

        await asyncio.create_task(
            process(
                checkpoint=checkpoint, 
                prompt=prompt, 
                negative_prompt=negative_prompt,
                batch_size=batch_size,
                height=height,
                width=width,
            )
        )
