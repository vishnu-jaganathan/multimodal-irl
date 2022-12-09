# https://www.assemblyai.com/blog/real-time-speech-recognition-with-python/

import pyaudio
import websockets
import asyncio
import base64
import json
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer


auth_key = '23ddd80c0fcf4eba92c73263e20eaa98'

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

FINAL_SCORE = None

# the AssemblyAI endpoint we're going to hit
URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

async def send_receive(stream):
   print(f'Connecting websocket to url ${URL}')
   async with websockets.connect(
       URL,
       extra_headers=(("Authorization", auth_key),),
       ping_interval=5,
       ping_timeout=20
   ) as _ws:
        await asyncio.sleep(0.1)
        print("Receiving SessionBegins ...")
        session_begins = await _ws.recv()
        print(session_begins)
        print("Sending messages ...")
        async def send():
            while True:
                try:
                    data = stream.read(FRAMES_PER_BUFFER)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data":str(data)})
                    await _ws.send(json_data)
                except Exception as e:
                    # this likely means receiver got that data
                    return None
                await asyncio.sleep(0.01)

            return None

        async def receive():
            prev_sentence = ''
            while True:
                try:
                    result_str = await _ws.recv()
                    sentence = json.loads(result_str)['text']
                    print(sentence)
                    if len(sentence) == 0 and len(prev_sentence) != 0:
                        sid = SentimentIntensityAnalyzer()
                        final_score = sid.polarity_scores(prev_sentence)['compound']
                        await _ws.close()
                        return final_score
                    prev_sentence = sentence
                except Exception as e:
                    return None

        return await asyncio.gather(send(), receive())


def get_nlp_score():

    
    p = pyaudio.PyAudio()
    
    # starts recording
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    send_result, receive_result = asyncio.run(send_receive(stream))
    print(receive_result)
    return receive_result


if __name__ == "__main__":
    get_nlp_score()
