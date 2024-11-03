import chess
import PySimpleGUI as sg
import chess.svg

def draw_board(board):
    return chess.svg.board(board=board, size=400)

def create_window(board):
    layout = [
        [sg.Image(data=draw_board(board), key='-BOARD-')],
        [sg.Text("Enter your move (e.g. e2e4): "), sg.InputText(key='-MOVE-')],
        [sg.Button('Make Move'), sg.Button('Reset')]
    ]
    return sg.Window('Chess Game', layout, finalize=True)

def main():
    board = chess.Board()
    window = create_window(board)

    while True:
        try:
            event, values = window.read()

            if event == sg.WIN_CLOSED:
                break

            if event == 'Make Move':
                move = values['-MOVE-']
                try:
                    move = chess.Move.from_uci(move)
                    if move in board.legal_moves:
                        board.push(move)
                        window['-BOARD-'].update(data=draw_board(board))
                        window['-MOVE-'].update('')  # Clear input
                    else:
                        sg.popup_error("Illegal move!")
                except:
                    sg.popup_error("Invalid move format!")

            if event == 'Reset':
                board.reset()
                window['-BOARD-'].update(data=draw_board(board))
        except any as e:
            print(e)

    window.close()

if __name__ == '__main__':
    main()