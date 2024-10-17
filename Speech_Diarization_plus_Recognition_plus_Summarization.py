import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from pyannote.audio import Pipeline 
import whisper 
from pydub import AudioSegment
from openai import OpenAI


app = FastAPI()

class SpeechProcessingPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = whisper.load_model("base").to(self.device)
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="your_huggingface_access_token"  # Replace with your actual token
        )
        self.pipeline.to(self.device)

    def diarize_and_transcribe(self, audio_file):
        diarization = self.pipeline({"audio": audio_file})
        audio = AudioSegment.from_wav(audio_file)
        transcriptions = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_audio = audio[turn.start * 1000: turn.end * 1000]
            segment_filename = f"temp_segment_{turn.start:.2f}_{turn.end:.2f}.wav"
            speaker_audio.export(segment_filename, format="wav")
            
            result = self.model.transcribe(segment_filename)
            transcription = result['text']
            transcriptions.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "text": transcription.strip()
            })
            os.remove(segment_filename)

        return transcriptions


class Summarizer: 
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = "your_openai_api_key_token"  # Replace with your actual API key

    def summarize_text(self, text) -> str: 
        # Note: You might need to implement the OpenAI client initialization as needed
        # This is a simplified version

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content


@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Initialize the pipeline
        pipeline = SpeechProcessingPipeline()
        summarizer = Summarizer()

        # Perform diarization and transcription
        transcription_results = pipeline.diarize_and_transcribe(file_location)

        # Format transcription results into a single text
        summarized_text = summarizer.summarize_text("\n".join(
            [result["text"] for result in transcription_results]
        ))

        # Cleanup temporary file
        os.remove(file_location)

        return JSONResponse(content={"summarized_text": summarized_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
