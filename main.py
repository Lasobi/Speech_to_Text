import argparse
import tkinter as tk
import sounddevice as sd
import soundfile as sf
import whisper
import pyperclip

class App:
    def __init__(self, master, args):
        self.master = master
        self.verbose = args.verbose
        master.title("Audio Recorder")

        self.record_button = tk.Button(master, text="Record", command=self.record_audio)
        self.record_button.pack()

        self.is_recording = False

    def record_audio(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.config(text="Stop")
            duration = 30
            if self.verbose:
                print(f"Recording started for {duration} seconds")
            self.recording = sd.rec(int(duration * 44100), samplerate=44100, channels=2)
        else:
            self.is_recording = False
            self.record_button.config(text="Record")
            sf.write("output.mp3", self.recording, 44100)
            model = whisper.load_model("base")

            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio("output.mp3")
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            if self.verbose:
                print(f"Detected language: {max(probs, key=probs.get)}")

            # decode the audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)
            if self.verbose:
                print(f"Result: {result.text}")

            # save to clipboard
            pyperclip.copy(result.text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()

    root = tk.Tk()
    root.minsize(width=200, height=20)
    app = App(root, args)
    root.mainloop()
