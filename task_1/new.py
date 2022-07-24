import os

from time import sleep
import vizdoom as vzd



import sys, random, math, pygame
from pygame.locals import *
from math import sqrt, cos, sin, atan2
import cv2
import numpy as np

# constants
XDIM = 637
YDIM = 480
WINSIZE = [XDIM, YDIM]
stepsize = 4.0
NUMNODES = 60000
RADIUS = 10

def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

def step_from_to(p1, p2):
    if dist(p1, p2) < stepsize:
        return p2
    else:
        theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
        return p1[0] + stepsize * cos(theta), p1[1] + stepsize * sin(theta)


def chooseParent(nn, newnode, nodes):
    for p in nodes:
        if dist([p.x, p.y], [newnode.x, newnode.y]) < RADIUS and p.cost + dist([p.x, p.y],
                                                                               [newnode.x, newnode.y]) < nn.cost + dist(
                [nn.x, nn.y], [newnode.x, newnode.y]):
            nn = p
    newnode.cost = nn.cost + dist([nn.x, nn.y], [newnode.x, newnode.y])
    newnode.parent = nn
    return newnode, nn


def reWire(nodes, newnode, pygame, screen):
    white = 255, 240, 200
    black = 20, 20, 40
    for i in range(len(nodes)):
        p = nodes[i]
        if p != newnode.parent and dist([p.x, p.y], [newnode.x, newnode.y]) < RADIUS and newnode.cost + dist([p.x, p.y],
                                                                                                             [newnode.x,
                                                                                                              newnode.y]) < p.cost:
            pygame.draw.line(screen, white, [p.x, p.y], [p.parent.x, p.parent.y])
            p.parent = newnode
            p.cost = newnode.cost + dist([p.x, p.y], [newnode.x, newnode.y])
            nodes[i] = p
            pygame.draw.line(screen, black, [p.x, p.y], [newnode.x, newnode.y])
    return nodes


def drawSolutionPath(start, goal, nodes, pygame, screen):
    pink = 200, 20, 240
    nn = nodes[0]
    for p in nodes:
        if dist([p.x, p.y], [goal.x, goal.y]) < dist([nn.x, nn.y], [goal.x, goal.y]):
            nn = p
    if dist([nn.x,nn.y],[goal.x,goal.y])>20:
        print("path not found")
        return
    while nn != start:
        path.append((nn.x,nn.y))
        pygame.draw.line(screen, pink, [nn.x, nn.y], [nn.parent.x, nn.parent.y], 5)
        nn = nn.parent

def isInObstacle(vex, obstacles):
    if vex.y < 0 or vex.y > 477:
        return True
    if vex.x < 0 or vex.x > 634:
        return True
    # for x,y in corners :
    #     if sqrt((vex.x-x)**2 + (vex.y-y)**2)<=2:
    #         return True
    alpha = math.floor(vex.y)
    beta = math.floor(vex.x)
    if (obstacles[alpha][beta] == 0 or obstacles[alpha + 1][beta] == 0 or obstacles[alpha - 1][beta] == 0 or
            obstacles[alpha][beta + 1] == 0 or obstacles[alpha][beta - 1] == 0 ):
        return True
    return False


def isThruObstacle(p0,p1, obstacles):
    xm = int((p0[0] + p1[0]) / 2)
    ym = int((p0[1] + p1[1]) / 2)
    if ym < 0 or ym >= 480:
        return True
    if xm < 0 or xm >= 637:
        return True
    if obstacles[ym][xm] == 0:
        return True
    xm1 = int((p0[0] + xm) / 2)
    ym1 = int((p0[1] + ym) / 2)
    if obstacles[ym1][xm1] == 0:
        return True
    xm2 = int((p1[0] + xm) / 2)
    ym2 = int((p1[1] + ym) / 2)
    if obstacles[ym2][xm2] == 0:
        return True
    if ((p1[0] - p0[0]) != 0):
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        # for x,y in corners :
        #     if(abs((y-p0[1]-m*(x-p0[0])))/(sqrt(1+m**2))<=2):
        #         return True
        step = 0
        if (p1[0] > p0[0]):
            step = 1
        else:
            step = -1
        xcoord = p0[0]
        ycoord = p0[1]
        i = 1
        while ((xcoord < p1[0] and step > 0) or (xcoord > p1[0] and step < 0)):
            xcoord += step
            ycoord += m
            if (isInObstacle(Node(xcoord, ycoord), obstacles)):
                return True
    else:
        step = 0
        if (p1[1] - p0[1] >= 0):
            step = 1
        else:
            step = -1
        ycoord = p0[1]
        while ((ycoord < p1[1] and step > 0) or (ycoord > p1[1] and step < 0)):
            ycoord += step
            if (isInObstacle(Node(p0[0], ycoord), obstacles)):
                return True
    return False


class Node:
    x = 0
    y = 0
    cost = 0
    parent = None

    def __init__(self, xcoord, ycoord):
        self.x = xcoord
        self.y = ycoord

def main():

    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('RRTstar')

    white = 255, 240, 200
    black = 20, 20, 40
    screen.fill(white)

    nodes = []


    nodes.append(Node(449., 214.))
    start = nodes[0]
    goal = Node(494., 367.)
    for i in range(NUMNODES):
        rand = Node(random.random() * XDIM, random.random() * YDIM)
        nn = nodes[0]
        for p in nodes:
            if dist([p.x, p.y], [rand.x, rand.y]) < dist([nn.x, nn.y], [rand.x, rand.y]):
                nn = p
        interpolatedNode = step_from_to([nn.x, nn.y], [rand.x, rand.y])


        newnode = Node(interpolatedNode[0], interpolatedNode[1])
        if isInObstacle(newnode,obstacles):
            continue
        [newnode, nn] = chooseParent(nn, newnode, nodes)
        if isThruObstacle((newnode.x,newnode.y),(nn.x,nn.y),obstacles) :
            continue

        nodes.append(newnode)
        pygame.draw.line(screen, black, [nn.x, nn.y], [newnode.x, newnode.y])
        nodes = reWire(nodes, newnode, pygame, screen)
        pygame.display.update()


        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Leaving because you requested it.")
    drawSolutionPath(start, goal, nodes, pygame, screen)
    for i in range(len(path) - 1):
        cv2.line(img, (int(path[i][0]), int(path[i][1])), (int(path[i + 1][0]), int(path[i + 1][1])), (0, 255, 255), 1)
    pygame.display.update()
    cv2.imshow("Final Path", img)
    cv2.imwrite("path.jpg",img)
    cv2.waitKey(0)

def f0(a,b) :
    if(abs(a-b)<5):
        return True
    return False
def f2(a,b) :
    if(abs(a-b)<5):
        return True
    return False

def f1(a,b) :
    if(abs(a-b)<0.5):
        return True
    return False


if __name__ == '__main__':
    img = cv2.imread("map_fu.png")

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path=[]

    blur = cv2.GaussianBlur(imgray, (3, 3), 0)
    ret, thresh = cv2.threshold(blur, 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    obstacles = thresh
    corner = cv2.goodFeaturesToTrack(obstacles, 200, 0.01, 10)
    corner = np.int0(corner)
    corners = []



    main()
    path.append((449., 214.))
    path.reverse()
    path.append((494., 367.))


    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



    game = vzd.DoomGame()


    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "s.wad"))


    game.set_doom_map("map01")


    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)


    game.set_screen_format(vzd.ScreenFormat.BGR24)




    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)


    game.set_objects_info_enabled(True)


    game.set_sectors_info_enabled(True)


    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)


    game.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,vzd.Button.MOVE_LEFT,vzd.Button.MOVE_RIGHT])

    print("Available buttons:", [b.name for b in game.get_available_buttons()])


    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    print("Available game variables:", [v.name for v in game.get_available_game_variables()])


    game.set_episode_timeout(2500)


    game.set_episode_start_time(10)


    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    game.set_window_visible(True)




    game.set_living_reward(-1)


    game.set_mode(vzd.Mode.PLAYER)




    game.init()
    if(len(path)==2):
        game.close()


    actions = [[True, False, False,False,False], [False, True, False,False,False], [False, False, True,False,False],[False, False, False,True,False],[False, False, False,False,True]]


    episodes = 1

    c = 10.99853611

    prev_angle=0
    prev_x=0
    prev_y=0
    prev2_x=0
    prev2_y=-64
    i=0


    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))


        game.new_episode()
        sum = 0


        x1=0
        y1=-64

        p=0

        for x,y in path:


            x=(x-449)*c
            y=(214-y)*c-64

            if(x==449 and y==214):
                x1=0
                y1=-64
                continue


            while f0(x1,x)==False or f0(y,y1)==False :

                state = game.get_state()


                n = state.number
                vars = state.game_variables
                screen_buf = state.screen_buffer
                depth_buf = state.depth_buffer
                labels_buf = state.labels_buffer
                automap_buf = state.automap_buffer
                labels = state.labels
                objects = state.objects
                sectors = state.sectors






                an= atan2(y-vars[2] , x-vars[1] )
                ang= math.degrees(an)
                if(ang<0):
                    ang= ang +360

                p = vars[1]
                q = vars[2]
                l=0
                r=0
                if vars[3]<ang:
                    l=ang-vars[3]
                    r=vars[3]+360-ang
                elif ang<vars[3] :
                    l=360-vars[3]+ang
                    r=vars[3]-ang

                if(f2(vars[3],ang)==False):
                    if(r<l):
                        game.set_action(actions[2])
                        tics = 1
                        update_state = True
                        game.advance_action(tics)
                    else :

                        game.set_action(actions[1])
                        tics = 1
                        update_state = True
                        game.advance_action(tics)





                elif f0(vars[1],x)==False or f0(vars[2],y)==False :
                    game.set_action(actions[0])
                    tics = 1
                    update_state = True
                    game.advance_action(tics)
                    automap = state.automap_buffer









                state1 = game.get_state()
                vars1= state1.game_variables
                x1=vars1[1]
                y1=vars1[2]



                automap = state.automap_buffer

                if automap is not None:
                    cv2.imshow('ViZDoom Map Buffer', automap)
                    cv2.waitKey(int(100 * sleep_time))


                print("State #" + str(n))
                print("Game variables:", vars)

                print("=====================")

                if sleep_time > 0:
                    sleep(sleep_time)


        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")


    game.close()