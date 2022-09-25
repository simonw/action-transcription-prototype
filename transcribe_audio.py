import sys
import replicate
import json


def transcribe_audio(filepath):
    model = replicate.models.get("cjwbw/whisper")
    return model.predict(
        audio=open(filepath, "rb"),
        model="large",
        translate=True
    )


if __name__ == "__main__":
    filepath = sys.argv[-1]
    if not filepath.endswith(".mp3"):
        print("Please provide an mp3 file")
        sys.exit(1)
    output = transcribe_audio(sys.argv[-1])
    print(json.dumps(output, indent=2))
