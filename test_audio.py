from src.audio_infer import predict_audio

audio = input("Enter audio path: ")

result = predict_audio(audio)

print("\n===== AUDIO RESULT =====")
print("Prediction:", result["label"])
print("Confidence:", result["confidence"], "%")
print("Real Probability:", result["real_probability"], "%")
print("Fake Probability:", result["fake_probability"], "%")