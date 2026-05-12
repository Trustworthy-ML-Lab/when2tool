from .base_env import BaseEnv


MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}
CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}

# Custom encoding: a user-defined substitution cipher (non-standard, impossible to guess)
CUSTOM_ENCODINGS = {
    "alpha7": {chr(i): chr(((i - 65 + 7) % 26) + 65) for i in range(65, 91)},
    "reverse": {chr(i): chr(90 - (i - 65)) for i in range(65, 91)},
    "scramble1": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "QWERTYUIOPASDFGHJKLZXCVBNM")),
    "scramble2": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "MXBFVZNLHTKGDWRJYPAQOIUCES")),
}
CUSTOM_DECODINGS = {name: {v: k for k, v in mapping.items()} for name, mapping in CUSTOM_ENCODINGS.items()}


class DecodingEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def _morse_decode(self, msg):
        words = msg.strip().split(" / ")
        decoded = []
        for word in words:
            letters = word.strip().split(" ")
            decoded.append("".join(MORSE_TO_CHAR.get(l, "?") for l in letters))
        return " ".join(decoded)

    def _morse_encode(self, msg):
        words = msg.upper().split(" ")
        encoded = []
        for word in words:
            encoded.append(" ".join(CHAR_TO_MORSE.get(c, "?") for c in word))
        return " / ".join(encoded)

    def _caesar_decode(self, msg, shift):
        out = []
        for c in msg:
            if c.isalpha():
                base = 65 if c.isupper() else 97
                out.append(chr((ord(c) - base - shift) % 26 + base))
            else:
                out.append(c)
        return "".join(out)

    def _caesar_encode(self, msg, shift):
        out = []
        for c in msg:
            if c.isalpha():
                base = 65 if c.isupper() else 97
                out.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                out.append(c)
        return "".join(out)

    def decode(self, encoded_message, encoding_type, key=None):
        etype = encoding_type.strip().lower()
        if etype == "morse":
            result = self._morse_decode(encoded_message)
        elif etype == "caesar":
            if key is None:
                return {"success": False, "message": "Caesar cipher requires a shift key."}
            result = self._caesar_decode(encoded_message, int(key))
        elif etype == "custom":
            if key is None or key not in CUSTOM_DECODINGS:
                return {"success": False, "message": f"Unknown custom encoding: {key}. Available: {list(CUSTOM_DECODINGS.keys())}"}
            mapping = CUSTOM_DECODINGS[key]
            result = "".join(mapping.get(c.upper(), c) for c in encoded_message)
        else:
            return {"success": False, "message": f"Unknown encoding: {encoding_type}"}
        return {"success": True, "decoded": result, "encoding_type": etype, "input_length": len(encoded_message), "output_length": len(result)}

    def encode(self, message, encoding_type, key=None):
        etype = encoding_type.strip().lower()
        if etype == "morse":
            result = self._morse_encode(message)
        elif etype == "caesar":
            if key is None:
                return {"success": False, "message": "Caesar cipher requires a shift key."}
            result = self._caesar_encode(message, int(key))
        elif etype == "custom":
            if key is None or key not in CUSTOM_ENCODINGS:
                return {"success": False, "message": f"Unknown custom encoding: {key}. Available: {list(CUSTOM_ENCODINGS.keys())}"}
            mapping = CUSTOM_ENCODINGS[key]
            result = "".join(mapping.get(c.upper(), c) for c in message.upper())
        else:
            return {"success": False, "message": f"Unknown encoding: {encoding_type}"}
        return {"success": True, "encoded": result, "encoding_type": etype, "input_length": len(message), "output_length": len(result)}
