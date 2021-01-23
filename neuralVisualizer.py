import pygame
pygame.init()

def drawNetwork(nn):
    screen_length = 800
    screen_width = 600
    screen = pygame.display.set_mode([screen_length,screen_width])

    #calculate the dimensioning for drawing the vertices
    length = 2 + len(nn.neuronList)
    height = len(nn.inList)
    
