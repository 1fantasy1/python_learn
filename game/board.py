import pygame
import sys
import numpy as np
import random
from pygame.locals import *

# 初始化
pygame.init()
pygame.display.set_caption('五子棋人机对战')

# 常量定义
BOARD_SIZE = 15  # 15x15的棋盘
GRID_SIZE = 40  # 每个格子的像素大小
PIECE_RADIUS = 18  # 棋子半径
MARGIN = 40  # 棋盘边距
WIDTH = BOARD_SIZE * GRID_SIZE + 2 * MARGIN
HEIGHT = BOARD_SIZE * GRID_SIZE + 2 * MARGIN
FPS = 30

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (220, 179, 92)
LINE_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 0, 0)

# 创建游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# 游戏状态
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)  # 0=空, 1=黑, 2=白
current_player = 1  # 1=玩家(黑), 2=电脑(白)
game_over = False
winner = None
last_move = None

# 评分表
score_table = {
    # 进攻分数
    "five": 100000,  # 五连
    "four": 10000,  # 活四
    "three": 1000,  # 活三
    "two": 100,  # 活二
    "one": 10,  # 活一
    "dead_four": 5000,  # 冲四
    "dead_three": 500,  # 眠三
    "dead_two": 50,  # 眠二

    # 防守分数 (比进攻略高)
    "opp_five": 1000000,
    "opp_four": 20000,
    "opp_three": 2000,
    "opp_two": 200,
    "opp_one": 20,
    "opp_dead_four": 10000,
    "opp_dead_three": 1000,
    "opp_dead_two": 100
}


def draw_board():
    """绘制棋盘"""
    screen.fill(BOARD_COLOR)

    # 绘制网格线
    for i in range(BOARD_SIZE):
        # 横线
        pygame.draw.line(screen, LINE_COLOR,
                         (MARGIN, MARGIN + i * GRID_SIZE),
                         (WIDTH - MARGIN, MARGIN + i * GRID_SIZE), 2)
        # 竖线
        pygame.draw.line(screen, LINE_COLOR,
                         (MARGIN + i * GRID_SIZE, MARGIN),
                         (MARGIN + i * GRID_SIZE, HEIGHT - MARGIN), 2)

    # 绘制天元和星位
    star_points = [3, 7, 11]  # 15路棋盘的星位
    for x in star_points:
        for y in star_points:
            pygame.draw.circle(screen, BLACK,
                               (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE), 5)


def draw_pieces():
    """绘制棋子"""
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 1:  # 黑棋
                pygame.draw.circle(screen, BLACK,
                                   (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE),
                                   PIECE_RADIUS)
            elif board[y][x] == 2:  # 白棋
                pygame.draw.circle(screen, WHITE,
                                   (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE),
                                   PIECE_RADIUS)

    # 高亮显示最后一步
    if last_move:
        x, y = last_move
        pygame.draw.circle(screen, HIGHLIGHT_COLOR,
                           (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE),
                           PIECE_RADIUS // 2, 2)


def draw_game_status():
    """显示游戏状态"""
    font = pygame.font.SysFont('Arial', 20)
    if game_over:
        if winner == 1:
            text = "游戏结束! 黑方获胜!"
        elif winner == 2:
            text = "游戏结束! 白方获胜!"
        else:
            text = "游戏结束! 平局!"
    else:
        text = "当前: " + ("黑方(你)" if current_player == 1 else "白方(电脑)")

    status_surface = font.render(text, True, BLACK)
    screen.blit(status_surface, (MARGIN, 10))


def get_board_position(mouse_pos):
    """将鼠标位置转换为棋盘坐标"""
    x, y = mouse_pos
    board_x = round((x - MARGIN) / GRID_SIZE)
    board_y = round((y - MARGIN) / GRID_SIZE)

    if 0 <= board_x < BOARD_SIZE and 0 <= board_y < BOARD_SIZE:
        return board_x, board_y
    return None


def is_valid_move(x, y):
    """检查落子是否有效"""
    return board[y][x] == 0


def make_move(x, y, player):
    """在指定位置落子"""
    global board, current_player, last_move
    board[y][x] = player
    last_move = (x, y)
    current_player = 3 - player  # 切换玩家 (1->2, 2->1)


def check_winner(x, y, player):
    """检查是否有玩家获胜"""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、斜

    for dx, dy in directions:
        count = 1  # 当前棋子

        # 正向检查
        nx, ny = x + dx, y + dy
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == player:
            count += 1
            nx += dx
            ny += dy

        # 反向检查
        nx, ny = x - dx, y - dy
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == player:
            count += 1
            nx -= dx
            ny -= dy

        if count >= 5:
            return True

    return False


def evaluate_line(line, player):
    """评估一行棋子的分数"""
    score = 0
    opp_player = 3 - player

    # 将数组转换为字符串便于匹配
    line_str = ''.join(map(str, line))

    # 检查五连
    if str(player) * 5 in line_str:
        return score_table["five"]

    # 检查对手五连
    if str(opp_player) * 5 in line_str:
        return -score_table["opp_five"]

    # 检查活四
    if line_str.count('0' + str(player) * 4 + '0') > 0:
        score += score_table["four"]

    # 检查冲四
    patterns = [
        str(player) * 4 + '0',  # 四连+空
        '0' + str(player) * 4,  # 空+四连
        str(player) + '0' + str(player) * 3,  # 中间断开的四连
        str(player) * 2 + '0' + str(player) * 2,
        str(player) * 3 + '0' + str(player)
    ]

    for pattern in patterns:
        if pattern in line_str:
            score += score_table["dead_four"]
            break

    # 检查活三
    if '0' + str(player) * 3 + '0' in line_str:
        score += score_table["three"]

    # 检查眠三
    patterns = [
        str(player) * 3 + '0',
        '0' + str(player) * 3,
        str(player) + '0' + str(player) * 2,
        str(player) * 2 + '0' + str(player)
    ]

    for pattern in patterns:
        if pattern in line_str:
            score += score_table["dead_three"]
            break

    # 检查活二
    if '0' + str(player) * 2 + '0' in line_str:
        score += score_table["two"]

    # 检查眠二
    if str(player) * 2 + '0' in line_str or '0' + str(player) * 2 in line_str:
        score += score_table["dead_two"]

    # 检查活一
    if '0' + str(player) + '0' in line_str:
        score += score_table["one"]

    # 同样的方式检查对手的棋型
    # 检查对手活四
    if line_str.count('0' + str(opp_player) * 4 + '0') > 0:
        score -= score_table["opp_four"]

    # 检查对手冲四
    patterns = [
        str(opp_player) * 4 + '0',
        '0' + str(opp_player) * 4,
        str(opp_player) + '0' + str(opp_player) * 3,
        str(opp_player) * 2 + '0' + str(opp_player) * 2,
        str(opp_player) * 3 + '0' + str(opp_player)
    ]

    for pattern in patterns:
        if pattern in line_str:
            score -= score_table["opp_dead_four"]
            break

    # 检查对手活三
    if '0' + str(opp_player) * 3 + '0' in line_str:
        score -= score_table["opp_three"]

    # 检查对手眠三
    patterns = [
        str(opp_player) * 3 + '0',
        '0' + str(opp_player) * 3,
        str(opp_player) + '0' + str(opp_player) * 2,
        str(opp_player) * 2 + '0' + str(opp_player)
    ]

    for pattern in patterns:
        if pattern in line_str:
            score -= score_table["opp_dead_three"]
            break

    # 检查对手活二
    if '0' + str(opp_player) * 2 + '0' in line_str:
        score -= score_table["opp_two"]

    # 检查对手眠二
    if str(opp_player) * 2 + '0' in line_str or '0' + str(opp_player) * 2 in line_str:
        score -= score_table["opp_dead_two"]

    # 检查对手活一
    if '0' + str(opp_player) + '0' in line_str:
        score -= score_table["opp_one"]

    return score


def evaluate_position(x, y, player):
    """评估在(x,y)位置落子的分数"""
    score = 0

    # 四个方向评估
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dx, dy in directions:
        # 获取一行9个棋子 (包括中心点)
        line = []
        for i in range(-4, 5):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                line.append(board[ny][nx])
            else:
                line.append(-1)  # 边界外

        score += evaluate_line(line, player)

    # 中心位置加分
    center = BOARD_SIZE // 2
    distance = abs(x - center) + abs(y - center)
    score += (BOARD_SIZE - distance) * 2

    return score


def computer_move():
    """电脑AI选择最佳落子位置"""
    best_score = -float('inf')
    best_moves = []

    # 遍历所有空位
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                # 评估这个位置的分数
                score = evaluate_position(x, y, 2)  # 电脑是白方(2)

                # 如果比当前最好分数高，更新最佳位置
                if score > best_score:
                    best_score = score
                    best_moves = [(x, y)]
                elif score == best_score:
                    best_moves.append((x, y))

    # 如果有多个最佳位置，随机选择一个
    if best_moves:
        return random.choice(best_moves)
    return None


def check_game_over():
    """检查游戏是否结束"""
    global game_over, winner

    # 检查棋盘是否已满
    if np.all(board != 0):
        game_over = True
        winner = None
        return

    # 检查最后一步是否导致胜利
    if last_move:
        x, y = last_move
        player = board[y][x]
        if check_winner(x, y, player):
            game_over = True
            winner = player


def reset_game():
    """重置游戏"""
    global board, current_player, game_over, winner, last_move
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    current_player = 1
    game_over = False
    winner = None
    last_move = None


# 主游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_r:  # 按R键重置游戏
                reset_game()
        elif event.type == MOUSEBUTTONDOWN and not game_over and current_player == 1:
            # 玩家回合
            pos = get_board_position(event.pos)
            if pos and is_valid_move(*pos):
                x, y = pos
                make_move(x, y, 1)
                check_game_over()

                # 电脑回合
                if not game_over and current_player == 2:
                    move = computer_move()
                    if move:
                        make_move(*move, 2)
                        check_game_over()

    # 绘制游戏
    draw_board()
    draw_pieces()
    draw_game_status()

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
