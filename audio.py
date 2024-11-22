from pygame import mixer

class ChessAudio():
    def __init__(self):
        mixer.init()

    def play_move(self):
        mixer.music.load("./audio/move-self.mp3")
        mixer.music.play()

    def play_capture(self):
        mixer.music.load("./audio/capture.mp3")
        mixer.music.play()