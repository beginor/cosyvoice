import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse


from os import path
from io import BytesIO
import torch

from .utils import (
    read_text_file,
    read_json_file,
    convert_to_wave_io,
    wave_to_mp3,
    SpeechModel
)

assets = './asset'

voices = f'{assets}/voices'
default_ref_audio = f'{voices}/man.wav'
default_ref_text = f'{voices}/man.wav.txt'

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

cosyvoice_model = CosyVoice2(
    './pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    fp16=False,
)

app = FastAPI()


@app.post('/v1/audio/speech', tags=['audio'])
def speech(params: SpeechModel):
    voice = params.voice
    ref_audio = default_ref_audio
    ref_text = read_text_file(default_ref_text)

    if path.exists(f'{voices}/{voice}.wav') and path.exists(f'{voices}/{voice}.wav.txt'):
        ref_audio = f'{voices}/{voice}.wav'
        ref_text = read_text_file(f'{voices}/{voice}.wav.txt')
    
    try:
        prompt_speech_16k = load_wav(ref_audio, 16000)
        prompt_text = ref_text

        bytes_io = BytesIO()

        for i, j in enumerate(cosyvoice_model.inference_zero_shot(
            tts_text=params.input,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_16k,
            stream=False
        )):
            # Save the generated audio to a temporary file
            tts_speech = j['tts_speech']
            torchaudio.save(
                bytes_io,
                tts_speech,
                cosyvoice_model.sample_rate,
                format='wav',
            )
            bytes_io.seek(0)
        
        if params.response_format == 'mp3':
            bytes_io = wave_to_mp3(bytes_io, cosyvoice_model.sample_rate)
        
        return StreamingResponse(
            bytes_io,
            media_type="audio/mp3" if params.response_format == 'mp3' else 'audio/wav',
        )

    except Exception as e:
        print(f'Error loading reference audio: {e}')
        return JSONResponse(
            status_code=500,
            content={'error': 'Failed to load reference audio.'}
        )
