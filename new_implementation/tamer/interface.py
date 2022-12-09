import os
import pygame

from .transcribe import get_nlp_score


class Interface:
    """ Pygame interface for training TAMER """

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        self.curr_text = ""

        self.screen = pygame.display.set_mode((200, 100))
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

    def get_scalar_feedback(self):
        """
        Get human input. 'W' key for positive, 'A' key for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        area = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    reward = -1.0
                    break
                elif event.key == pygame.K_2:
                    reward = -0.8
                    break
                elif event.key == pygame.K_3:
                    reward = -0.6
                    break
                elif event.key == pygame.K_4:
                    reward = -0.4
                    break
                elif event.key == pygame.K_5:
                    reward = -0.2
                    break
                elif event.key == pygame.K_6:
                    reward = 0.2
                    break
                elif event.key == pygame.K_7:
                    reward = 0.4
                    break
                elif event.key == pygame.K_8:
                    reward = 0.6
                    break
                elif event.key == pygame.K_9:
                    reward = 0.8
                    break
                elif event.key == pygame.K_0:
                    reward = 1.0
                    break
                elif event.key == pygame.K_v:
                    reward = get_nlp_score()
                    break
        pygame.display.update(area)
        return reward

    def show_action(self, action):
        """
        Show agent's action on pygame screen
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)
