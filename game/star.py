import pygame
import random
import sys

# 初始化pygame
pygame.init()

# 屏幕设置
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("接大小不同的星星")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
COLORS = [(255, 255, 0), (255, 165, 0), (255, 192, 203), (255, 0, 255)]  # 黄色、橙色、粉色、紫色

# 玩家设置
player_width = 100
player_height = 20
player_x = WIDTH // 2 - player_width // 2
player_y = HEIGHT - 50
player_speed = 8

# 星星设置
stars = []  # 每个星星格式: [x, y, size, speed, color]
star_spawn_rate = 30  # 每30帧生成一个星星

# 分数
score = 0
font = pygame.font.SysFont(None, 36)

# 游戏循环
clock = pygame.time.Clock()
frame_count = 0


def draw_player(x, y):
    pygame.draw.rect(screen, BLUE, (x, y, player_width, player_height))


def draw_star(x, y, size, color):
    pygame.draw.polygon(screen, color, [
        (x, y - size // 2),
        (x + size // 4, y - size // 4),
        (x + size // 2, y),
        (x + size // 4, y + size // 4),
        (x, y + size // 2),
        (x - size // 4, y + size // 4),
        (x - size // 2, y),
        (x - size // 4, y - size // 4)
    ])


def spawn_star():
    size = random.randint(15, 40)  # 星星大小在15到40之间
    x = random.randint(size, WIDTH - size)
    y = -size
    # 星星越大，下落速度越慢，但分值越高
    speed = random.uniform(1.0, 5.0) * (40 / size)  # 大小与速度成反比
    color = random.choice(COLORS)
    stars.append([x, y, size, speed, color])


def update_stars():
    global score
    for star in stars[:]:
        star[1] += star[3]  # 使用星星自己的速度

        # 检查是否接到星星
        if (player_x < star[0] < player_x + player_width and
                player_y < star[1] + star[2] // 2 < player_y + player_height):
            stars.remove(star)
            # 星星越大，得分越高 (5-20分)
            score += int(star[2] / 3)  # 15-40大小对应5-13分

        # 检查星星是否超出屏幕底部
        elif star[1] > HEIGHT + star[2]:
            stars.remove(star)


running = True
while running:
    frame_count += 1

    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 玩家移动
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_width:
        player_x += player_speed

    # 生成星星
    if frame_count % star_spawn_rate == 0:
        spawn_star()

    # 更新星星位置
    update_stars()

    # 绘制
    screen.fill(BLACK)
    draw_player(player_x, player_y)
    for star in stars:
        draw_star(star[0], star[1], star[2], star[4])

    # 显示分数
    score_text = font.render(f"分数: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
