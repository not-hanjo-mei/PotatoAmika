import logging, sys, os
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from genie_tts.Converter.v2ProPlus.Converter import convert
base = os.environ.get('TTS_ENGINE_DIR', '.')  # working dir holding weights/ and graphs/
convert(f'{base}/weights/MekaHime_Amika.ckpt', f'{base}/weights/MekaHime_Amika.pth', f'{base}/graphs/fp32')
print('CONVERT_DONE')
