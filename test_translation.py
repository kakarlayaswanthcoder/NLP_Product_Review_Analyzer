from translator import translate_text


text = "This phone has an excellent camera and amazing battery life."


telugu = translate_text(text, "te")
hindi = translate_text(text, "hi")


print("English:")
print(text)

print("\nTelugu:")
print(telugu)

print("\nHindi:")
print(hindi)