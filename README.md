# инструкция
нужные библиотеки
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
```
pip install langdetect pillow disnake transformers 
```

как импортировать Cog смотрите тут - https://docs.disnake.dev/en/stable/ext/commands/cogs.html

как работать с web ui смотрите сдеть - https://github.com/AUTOMATIC1111/stable-diffusion-webui

чтобы запустить web ui с API в папке `webui` найдите `webui-user.bat`, откройте в редакторе и строчку `set COMMANDLINE_ARGS` замените на
```
set COMMANDLINE_ARGS=--api
```
