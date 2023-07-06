"""
Audio Recorder Application

This script implements a simple graphical user interface (GUI) for recording audio and performing text processing tasks.
It uses the tkinter library for GUI functionality, sounddevice and soundfile for audio recording and playback,
whisper for language detection and audio decoding, and pyperclip for copying the result to the clipboard.

The main class `App` represents the application and provides methods for recording audio, stopping the recording,
and performing text processing tasks on the recorded audio. The application window displays a 'Record' button that
triggers the recording process. When clicked again, it stops the recording, saves the audio to a file, performs
language detection, audio decoding, and copies the result to the clipboard.

The script can be executed from the command line with optional arguments:
    --verbose: Enable verbose output

Example:
    python audio_recorder.py --verbose
"""

import argparse
import tkinter as tk
import sounddevice as sd
import soundfile as sf
import whisper
import pyperclip


class App:
    def __init__(self, master, args):
        """
        Initializes the Audio Recorder application.

        Args:
            master (tkinter.Tk): The root window of the application.
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.master = master
        self.verbose = args.verbose
        master.title("Audio Recorder")

        self.record_button = tk.Button(master, text="Record", command=self.record_audio)
        self.record_button.pack()

        self.is_recording = False

    def record_audio(self):
        """
        Records audio when the 'Record' button is clicked.
        Stops recording and performs text processing when clicked again.
        """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    root = tk.Tk()
    root.minsize(width=200, height=20)
    app = App(root, args)
    root.mainloop()
