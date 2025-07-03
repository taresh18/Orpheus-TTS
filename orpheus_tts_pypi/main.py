from flask import Flask, Response, request
import struct
from orpheus_tts import OrpheusModel

app = Flask(__name__)
engine = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft", max_model_len=2048, gpu_memory_utilization=0.9, max_num_batched_tokens=8192, max_num_seqs=4, enable_chunked_prefill=True)

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,             
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
    voice = request.args.get('voice', 'tara')

    def generate_audio_stream():
        yield create_wav_header()

        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice=voice,
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=2000,
            temperature=0.4,
            top_p=0.9
        )
        for chunk in syn_tokens:
            yield chunk

    return Response(generate_audio_stream(), mimetype='audio/wav')

@app.route('/v1/audio/speech/stream', methods=['POST'])
def tts_stream():
    data = request.get_json()
    if not data:
        return Response("No JSON data provided.", status=400, mimetype='text/plain')

    prompt = data.get('input', 'Hey there, looks like you forgot to provide a prompt!')
    voice = data.get('voice', 'tara')

    def generate_audio_stream():
        # Parameters from the request are used, with defaults from the existing /tts endpoint.
        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice=voice,
            repetition_penalty=data.get('repetition_penalty', 1.1),
            stop_token_ids=data.get('stop_token_ids', [128258]),
            max_tokens=data.get('max_tokens', 2000),
            temperature=data.get('temperature', 0.4),
            top_p=data.get('top_p', 0.9)
        )
        for chunk in syn_tokens:
            yield chunk

    return Response(generate_audio_stream(), mimetype='audio/pcm')


VOICE_DETAILS = [
    {
        "name": "tara",
        "description": "A warm, friendly female voice with natural expressiveness",
        "language": "en",
        "gender": "female",
        "accent": "american",
        "preview_url": None
    },
    {
        "name": "zoe",
        "description": "A bright, cheerful female voice with expressive delivery",
        "language": "en",
        "gender": "female",
        "accent": "american",
        "preview_url": None
    },
    {
        "name": "jess",
        "description": "A warm, friendly female voice with expressive delivery",
        "language": "en",
        "gender": "female",
        "accent": "american",
        "preview_url": None
    },
    {
        "name": "zac",
        "description": "A warm, friendly male voice with expressive delivery",
        "language": "en",
        "gender": "male",
        "accent": "american",
        "preview_url": None
    }
]


@app.route("/api/voices", methods=['GET'])
def get_voices():
    """Get available voices with detailed information."""
    
    return {
        "voices": VOICE_DETAILS,
        "default": 'tara',
        "count": len(VOICE_DETAILS)
    }



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, threaded=True)
