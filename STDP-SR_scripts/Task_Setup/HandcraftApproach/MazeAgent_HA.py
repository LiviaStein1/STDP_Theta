from matplotlib import patches
import numpy as np 

import pandas as pd 
from torch.mtia import init
from tqdm.notebook import tqdm
from datetime import datetime 
import numbers
from pprint import pprint as pprintq
import os
from scipy.stats import vonmises
from scipy.spatial import distance_matrix
import dill 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import imageio 
from collections import deque
import matplotlib.animation as animation
from PIL import Image

import tomplotlib.tomplotlib as tpl
tpl.figureDirectory = './figures'
tpl.set_colorscheme(colorscheme=2)



#Default parameters for MazeAgent 
defaultParams = { 

          #Maze params 
          'mazeType'            : 'oneRoom',  #type of maze, define in getMaze() function
          'stateType'           : 'gaussian', #feature on which to TD learn (onehot, gaussian, gaussianCS, circles, bump)
          'movementPolicy'      : 'raudies',  #movement policy (raudies, random walk, windows screensaver)
          'roomSize'            : 1,          #maze size scaling parameter, metres
          'dt'                  : None,       #simulation time disretisation (defualts to largest )
          'dx'                  : 0.01,       #space discretisation (for plotting, movement is continuous)
          'speedScale'          : 0.16,       #movement speed scale, metres/second
          'rotSpeedScale'       : None,       #rotational speed scale, radians/second
          'initPos'             : [0.1,0.1],  #initial position [x0, y0], metres
          'initDir'             : [1,0],      #initial direction, unit vector
          'nCells'              : None,       #how many features to use
          'centres'             : None,       #array of receptive field positions. Overwrites nCells
          'sigma'               : 1,          #basis cell width scale (irrelevant for onehots)
          'doorsClosed'         : True,       #whether doors are opened or closed in multicompartment maze
          'reorderCells'        : True,       #whether to reorde the cell centres which have been provided
          'firingRateLookUp'    : False,      #use quantised lookup table for firing rates 
          'biasDoorCross'       : False,      #if True, in twoRoom maze door crossings are biased towards
          'biasWallFollow'      : True,       #if True, agent aligns to wall when gets too near.
          'biasToGoal'          : False,      #if True, agent gets a bias towards the goal location which increases with proximity to goal

          #TD params 
          'tau'                 : 4,          #TD decay time, seconds
          'TDdx'                : 0.01,       #rough distance between TD learning updates, metres 
          'alpha'               : 0.01,       #TD learning rate 
          'successorFeatureNorm': 100,        #linear scaling on successor feature definition found to improve learning stability
          'TDreg'               : 0.01,       #L2 regularisation 
          
          #STDP params
          'peakFiringRate'      : 5,          #peak firing rate of a cell (middle of place field,preferred theta phase)
          'tau_STDP_plus'       : 20e-3,      #pre trace decay time
          'tau_STDP_minus'      : 40e-3,      #post trace decay time
          'a_STDP'              : -0.4,       #pre-before-post potentiation factor (post-before-pre = 1) 
          'eta'                 : 0.05,       #STDP learning rate
          'baselineFiringRate'  : 0,          #baseline firing rate for cells 
          'use_full_STDP_rule'  : False,      #whether to use full STDP rule     
          'online_mapping'      : 'identity',  #how to map CA3-->CA1 during learning
          'rownorm'             : False,
            


          #Theta precession params
          'thetaFreq'           : 10,         #theta frequency
          'precessFraction'     : 0.5,        #fraction of 2pi the prefered phase moves through
          'kappa'               : 1,          # von mises spread parameter

          #Theta scrambling params (Livi addition)
          'scramble_strength'    : 0.5,        # strength of theta phase scrambling 
          'hf_strength'          : 0.5,        # strength of high frequency jitter added to scrambled theta phase

          # Goal Location
          'goalPos'             : None, # position of goal in metres -> given as array of shape [x, y]
          'goalRadius'          : None,       # radius of goal in metres
          'goalReward'          : None,        # reward given when goal is reached

}

class MazeAgent():
    """MazeAgent defines an agent moving around a maze. 
    The agent moves according to a predefined movement policy
    As the agent moves it learns a successor representation over state vectors according to a TD learning rule 
    The movement polcy is 
        (i)  continuous in space. There is no discretisation of location. Time is discretised into steps of dt
        (ii) completely decoupled from the TD learning.
    TD learning is 
        (i)  state general. i.e. it learns generic SRs for feature vectors which are not necessarily onehot. See de Cothi and Barry, 2020  
        (ii) time continuous. Defined in terms of a memory decay time tau, not unitless gamma. Any two states can be used fro a TD learning step irrespective of their seperation in time. 
    As the rat moves and learns its position and time stamps are continually saved. Periodically a snapshot of the current SR matrix and state of other parameters in the maze are also saved. 
    """   
    def __init__(self,
                params={},
                loadFromFileCalled=None):
        """Sets the parameters of the maze anad agent (using default if not provided) 
        and initialises everything. This includes: 
        •initilising history dataframes
        •making the maze (a dictionary of "walls" which cant be crossed)
        •setting position, velocity, time
        •discretising space into coordinates for later plotting
        •initialising basis features (gaussian centres, fourier frequencies etc.)
        •initialising SR matrix 

        Args:
            params (dict, optional): A dictionary of parameters which you want to differ from the default. Defaults to {}.
        """        
        if loadFromFileCalled is not None: 
            self.loadFromFile(name=loadFromFileCalled)
            
        else:
            print("Setting parameters")
            for key, value in defaultParams.items():
                setattr(self, key, value)
            self.updateParams(params)

            print("Initialising")
            self.initialise()
            print("DONE")

    def updateParams(self,
                     params : dict):        
        """Updates parameters from a dictionary. 
        All parameters found in params will be updated to new value

        Args:
            params (dict): dictionary of parameters to change
            initialise (bool, optional): [description]. Defaults to False.
        """        
        for key, value in params.items():
            setattr(self, key, value)

    def initialise(self): #should only be called once at the start 
        """Initialises the maze and agent. Should only be called once at the start.
        """        
        #initialise history dataframes
        print("   making state/history dataframes")
        self.mazeState = {}
        self.history = pd.DataFrame(columns = ['t','pos','delta','runID']) 
        self.snapshots = pd.DataFrame(columns = ['t','M','W','mazeState'])
        self.spikedata = {'CA3':{'times':[],'ids':[]}, 'CA1':{'times':[],'ids':[]}}
        self.decoded_dir_history = [] # to store the decoded direction at each time step for later plotting

        # new condition specific spike data (Livi addition)
        self.spikedata_by_condition = {
            'theta': {'CA1': {'times': [], 'ids': []}, 'CA3': {'times': [], 'ids': []}},
            'notheta': {'CA1': {'times': [], 'ids': []}, 'CA3': {'times': [], 'ids': []}},
            'scrambled': {'CA1': {'times': [], 'ids': []}, 'CA3': {'times': [], 'ids': []}}
        }

        #set pos/vel
        print("   initialising velocity, position and direction")
        self.pos = np.array(self.initPos)
        self.speed = self.speedScale
        self.dir = np.array(self.initDir)

        #time and runID
        print("   setting time/run counters")
        self.t = 0
        self.runID = 0  
        
        # initialise theta phase in radians -> to convert to degrees: radians * (180/np.pi) 
        print("   initialising theta phases: clean and scrambled")
        self.thetaPhase = self.thetaFreq*(self.t%(1/self.thetaFreq))*2*np.pi # has implicit modulo of 2pi, so always between 0 and 2pi
        # Livi addition: initialise scrambled theta phase to zero at time zero, will then be updated according to scrambling parameters during movement policy update
        self.thetaPhase_scrambled = 0.0 # single scalar value for current time

        # save history of scrambled theta phases for later plotting (Livi addition)
        self.thetaPhase_scrambled_history = [] # list to store scrambled theta phase at each time step
        self.thetaPhase_scrambled_time_history = [] # matching timestamps for scrambled phase values
        self._scrambled_phase_fallback_warned = False

        #make maze 
        print("   making the maze walls")
        self.walls = getWalls(mazeType=self.mazeType, roomSize=self.roomSize)
        walls = self.walls.copy()
        if self.doorsClosed == False: 
            del walls['doors']
            self.mazeState['walls'] = walls
        elif self.doorsClosed == True: 
            self.mazeState['walls'] = walls
        
        # Set Goal
        if (self.goalPos is not None) and (self.goalRadius is not None) and (self.goalReward is not None):
            print("   setting goal location and reward")
            self.setGoal(goalPos=self.goalPos, goalRadius=self.goalRadius, goalReward=self.goalReward)

        #extent, xArray, yArray, discreteCoords
        print("   discretising position for later plotting")
        if abs((self.roomSize / self.dx) - round(self.roomSize / self.dx)) > 0.00001:
            print("      dx must be an integer fraction of room size, setting it to %.4f, %g along room length" %(self.roomSize / round(self.roomSize / self.dx), round(self.roomSize / self.dx)))
            self.dx = self.roomSize / round(self.roomSize / self.dx)
        minx, maxx, miny, maxy = 0, 0, 0, 0
        for room in self.walls:
            wa = self.walls[room]
            minx, maxx, miny, maxy = min(minx,np.min(wa[...,0])), max(maxx,np.max(wa[...,0])), min(miny,np.min(wa[...,1])), max(maxy,np.max(wa[...,1])) 
        self.extent = np.array([minx,maxx,miny,maxy])
        self.width = maxx-minx
        self.height = maxy-miny
        self.xArray = np.arange(minx + self.dx/2, maxx, self.dx)
        self.yArray = np.arange(miny + self.dx/2, maxy, self.dx)[::-1]
        x_mesh, y_mesh = np.meshgrid(self.xArray,self.yArray)
        coordinate_mesh = np.array([x_mesh, y_mesh])
        self.discreteCoords = np.swapaxes(np.swapaxes(coordinate_mesh,0,1),1,2) #an array of discretised position coords over entire map extent 
        self.mazeState['extent'] = self.extent

        #handle None params
        print("   handling undefined parameters")
        if self.dt == None: 
            self.dt = min(self.tau_STDP_plus,self.tau_STDP_minus) / 2
        if self.pos is None: 
            ex = self.extent
            self.pos = np.array([ex[0] + 0.2*(ex[1]-ex[0]),ex[2] + 0.2*(ex[3]-ex[2])])
        if self.dir is None: 
            if self.mazeType == 'longCorridor': self.dir = np.array([0,1])
            elif self.mazeType == 'loop': self.dir = np.array([1,0])
            else: self.dir = np.array([1,1]) / np.sqrt(2)
        if self.rotSpeedScale is None: 
            if self.mazeType == 'loop' or self.mazeType == 'longCorridor':
                self.rotSpeedScale = np.pi
            else: 
                self.rotSpeedScale = 3*np.pi
        if (self.nCells is None) and (self.centres is None): 
            ex = self.extent
            area, pcarea  = (ex[1]-ex[0])*(ex[3]-ex[2]), np.pi * ((self.sigma/2)**2)
            cellsPerArea = 10
            self.nCells = int(cellsPerArea * area / pcarea) #~10 in any given place
        if self.mazeType == 'TMaze':
            self.LRDecisionPending=True
            self.lastArmChoice = 1 #1 for right, -1 for left. Only relevant if LRDecisionPending is True
        self.doorPassage = False
        self.doorPassageTime = 0
        self.reachedGoal = False
        self.goalReachedTime = 0
        self.lastTurnUpdate = -1
        self.randomTurnSpeed = 0

        #initialise basis cells and M (successor matrix)
        print("   initialising basis features for learning")

        if self.stateType in ['gaussian', 'gaussianCS','gaussianThreshold', 'circles','onehot','bump']:
            if self.centres is not None: #if we don't provide locations for cell centres...
                self.nCells = self.centres.shape[0]
                self.stateSize = self.nCells
            else: #scatter some ourselves (making sure they aren't too close)
                self.stateSize=self.nCells
                xcentres = np.random.uniform(self.extent[0],self.extent[1],self.nCells)
                ycentres = np.random.uniform(self.extent[2],self.extent[3],self.nCells)
                self.centres = np.array([xcentres,ycentres]).T
                inds = self.centres[:,0].argsort()
                self.centres = self.centres[inds]
                print("   checking basis cells aren't too close") 
                min_d = 0.1/0.9
                done = False
                while done != True:
                    min_d *= 0.9
                    print("     min seperation distance:  %.1f cm" %(min_d*100))
                    count = 0
                    while count <= 10:
                        d = distance_matrix(self.centres,self.centres)
                        d  += 0.1*np.eye(d.shape[0])
                        d_xid, d_yid = np.where(d < min_d)
                        print('      ',int(len(d_xid)/2),' overlapping pairs',end='\n')
                        if len(d_xid) == 0:
                            done = True 
                            break
                        to_remove = []
                        for i in range(len(d_xid)):
                            if d_xid[i] < d_yid[i]:
                                to_remove.append(d_xid[i])
                        to_remove = np.unique(to_remove)
                        xcentres = np.random.uniform(self.extent[0],self.extent[1],len(to_remove))
                        ycentres = np.random.uniform(self.extent[2],self.extent[3],len(to_remove))
                        self.centres[to_remove] = np.array([xcentres,ycentres]).T
                        count += 1
            self.M = np.eye(self.stateSize)
            self.W = self.M.copy() / self.nCells
            self.M_theta = self.M.copy()
            self.W_notheta = self.W.copy()
            self.W_scrambled = self.W.copy() #Livi addition scramble
            # initialise scrambled theta phase (Livi addition)
            # (old version of theta scrambled)  self.thetaPhase_scrambled = np.random.uniform(0,2*np.pi, size = self.nCells) # give scrambled condition random theta phase (Livi addition)



            #order the place cells so successor matrix has some structure:
            if self.reorderCells==True:
                if self.mazeType == 'twoRooms': #from centre outwards
                    middle = np.array([self.extent[1]/2,self.extent[3]/2])
                    distance_to_centre = np.linalg.norm(middle - self.centres,axis=1)
                    distance_to_centre = distance_to_centre * (2*(self.centres[:,0]>middle[0])-1)
                    inds = distance_to_centre.argsort()
                    self.centres = self.centres[inds]
                if self.mazeType == 'oneRoom': # from centre outwards, but with left and right halves ordered separately so that we can see the block structure in W more clearly
                    middle = np.array([self.extent[1]/2,self.extent[3]/2])
                    distance_to_centre = np.linalg.norm(middle - self.centres,axis=1)
                    distance_to_centre = distance_to_centre * (2*(self.centres[:,0]>middle[0])-1)
                    inds = distance_to_centre.argsort()
                    self.centres = self.centres[inds]
                else: #from left to right
                    inds = self.centres[:,0].argsort()
                    self.centres = self.centres[inds]

        elif self.stateType == 'fourier':
            self.stateSize = self.nCells
            self.kVectors = np.random.rand(self.nCells,2) - 0.5
            self.kVectors /= np.linalg.norm(self.kVectors, axis=1)[:,None]
            self.kFreq = 2*np.pi / np.random.uniform(0.01,1,size=(self.nCells))
            self.phi = np.random.uniform(0,2*np.pi,size=(self.nCells))
            self.M = np.eye(self.stateSize)
            #self.M = np.zeros((self.stateSize,self.stateSize))
        
        if hasattr(self.sigma,"__len__"):
            if self.sigma.__len__() == self.nCells:
                self.sigmas = self.sigma
        else:
            self.sigmas = np.array([self.sigma]*self.nCells)

        #array of states, one for each discretised position coordinate 
        print("   calculating state vector at all discretised positions")
        self.statesAlreadyInitialised = False
        self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords,stateType=self.stateType,verbose=True) #an array of discretised position coords over entire map extent 
        self.statesAlreadyInitialised = True

        #store time zero snapshot # Livi: added W_scrambled
        snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'W': [self.W.copy()],'W_notheta': [self.W_notheta.copy()], 'W_scrambled' : [self.W_scrambled.copy()], 'mazeState':[self.mazeState]})
        self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)

        #STDP stuff
        print("   initialising STDP weight matrix and traces")
        self.preTrace = np.zeros(self.nCells) #causes potentiation 
        self.preTrace_notheta = np.zeros(self.nCells) #causes potentiation 
        self.postTrace = np.zeros(self.nCells) #causes depression 
        self.postTrace_notheta = np.zeros(self.nCells) #causes depression
        self.lastSpikeTime = np.array(-10.0)
        self.lastSpikeTime_notheta = np.array(-10.0)
        self.spikeCount = np.array(0)
        self.spikeCount_notheta = np.array(0)
        # Livi: added scrambled condition
        print("   Adding 'theta scrambled' Condition")
        self.preTrace_scrambled = np.zeros(self.nCells) #for potentiation
        self.postTrace_scrambled = np.zeros(self.nCells) # for depression
        self.lastSpikeTime_scrambled = np.array(-10.0) # initialise time since last spike of any neuron (large bcs first)
        self.spikeCount_scrambled = np.array(0) # keep track of spike count (initialised as zero logically) 

        # initialise Value function -> value of cells defined by distance to goal
        print("   calculating value function based on distance to goal")
        if (self.goalPos is not None) and (self.goalRadius is not None) and (self.goalReward is not None):
            self.cellValues = self.getCellValues()
        else: 
            print("WARNING: Goal position, radius and reward NOT set! Value function initialised with 1s for every cell.")
            self.cellValues = np.ones(self.nCells) # to prevent that error at runtime when goal is not set or for older agents
        

    def runRat(self,
            trainTime=10,
            saveEvery=0.5,
            TDSRLearn=True,
            STDPLearn=True):
        """The main experiment call.
        A "run" consists of a period where the agent explores the maze according to the movement policy. 
        As it explores it learns, by TD, a successor representation over state vectors. 
        The can be called multiple times. Each successive run will be saved in self.history with an increasing runID
        Snapshots of the current SR matrix and mazeState can be saved along the way
        Runs can be interrupted with KeyboardInterrupt, data will still be saved. 
        Args:
            trainTime (int, optional): How long to explore in minutes. Defaults to 10.
            saveEvery (int, optional): Frequency to save snapshots, in minutes. Defaults to 1.
            TDSRLearn (bool,optional): toggles whether to do TD learning 
            STDPLearn (bool, optional): toggles whether to do STDP learning 
        """        
        steps = int(trainTime * 60 / self.dt) #number of steps to perform 

        hist_t = np.zeros(steps)
        hist_pos = np.zeros((steps,2))
        hist_delta = np.zeros(steps)

        lastTDstep, distanceToTD = 0, np.random.exponential(self.TDdx) #2cm scale
        
        """Main training loop. Principally on each iteration: 
            • always updates motion policy
            • often does TD learning step

            • sometimes saves snapshot"""
        for i in tqdm(range(steps)): #main training loop

            try:
                #update pos, velocity, direction and time according to movement policy
                self.movementPolicyUpdate()
                if i > 1:

                    # print(self.pos)
                    """STDP learning step"""
                    if (STDPLearn == True) and (self.stateType in  ['bump','gaussian', 'gaussianCS','gaussianThreshold', 'circles']):
                        if self.use_full_STDP_rule == True:
                            _ = self.STDPLearningStep_detailed(dt = self.t - hist_t[i-1])
                        else:
                            _ = self.STDPLearningStep(dt = self.t - hist_t[i-1])

                            
                    """TD learning step"""
                    if TDSRLearn == True: 
                        
                        alpha = self.alpha
                        try: alpha_ = alpha[0] * np.exp(-(i/steps)*(np.log(self.alpha[0]/self.alpha[1]))) #decaying alpha
                        except: alpha_ = self.alpha
                        

                        if np.linalg.norm(self.pos - hist_pos[lastTDstep]) >= distanceToTD: #if it's moved over 2cm meters from last step 
                            dtTD = self.t - hist_t[lastTDstep]
                            delta = self.TDLearningStep(pos=self.pos, prevPos=hist_pos[lastTDstep], dt=dtTD, tau=self.tau, alpha=alpha_)
                            lastTDstep = i 
                            distanceToTD = np.random.exponential(self.TDdx)
                            hist_delta[i] = delta


                # update clean theta phase but not scrambled theta phase 
                self.thetaPhase = self.thetaFreq*(self.t%(1/self.thetaFreq))*2*np.pi #8Hz theta 

                #update history arrays
                hist_pos[i] = self.pos
                hist_t[i] = self.t

                #save snapshot 
                if (isinstance(saveEvery, numbers.Number)) and (i % int(saveEvery * 60 / self.dt) == 0):
                    snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta':[self.W_notheta.copy()], 'W_scrambled':[self.W_scrambled.copy()], 'mazeState':[self.mazeState]}) #Livi: added scrambled
                    self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)

            except KeyboardInterrupt: 
                print("Keyboard Interrupt:")
                break
            # except ValueError as error:
            #     print("ValueError:")
            #     print(error)
            #     print(f"   Rat position: {self.pos}")
            #     break

        self.runID += 1
        runHistory = pd.DataFrame({'t':list(hist_t[:i]), 'pos':list(hist_pos[:i]),'delta':list(hist_delta[:i])})
        self.history = pd.concat([self.history, runHistory], ignore_index=True)
        snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta':[self.W_notheta.copy()], 'W_scrambled':[self.W_scrambled.copy()], 'mazeState':[self.mazeState]}) # Livi: added scrambled
        self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)

        #find and save grid/place cells so you don't have to repeatedly calculate them when plotting 
        print("Calculating place and grid cells")
        self.gridFields = self.getGridFields(self.M)
        self.placeFields = self.getPlaceFields(self.M)

        if TDSRLearn == True: 
            # plotter = Visualiser(self)
            # plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)
            delta = np.array(hist_delta)
            time = np.array(hist_t)
            time = time[delta!=0] / 60
            delta = delta[delta!=0]
            time, delta = time[::10], delta[::10]
            smooth_delta = [np.mean(delta[max(0,i-100):min(i+100,len(delta))]) for i in range(len(delta))]
            fig, ax = plt.subplots(figsize=(2,1))
            ax.scatter(time,delta,s=0.5,alpha=0.5)
            ax.scatter(time,smooth_delta,s=1,alpha=0.5,c='C2')
            ax.set_xlabel("Time / min")
            ax.set_ylabel("Update size")
    
    def runDecisionTask(self,
                        runTime=10, 
                        saveEvery=0.5, 
                        TDSRLearn=True, 
                        STDPLearn=True, 
                        condition = 'theta', 
                        phase_threshold = 1.5*np.pi, 
                        t_window =None,
                        updateSTDPWeights = False,
                        maxTurnAngle = False,
                        maxAngle = None):
        """
        Runs the decision task: The agent runs in the maze, starting at the stem for a certain amount of time.
        During the run the agents movement is defined by the direction encoded in the late phase spikes of the recent theta sweeps. 
        At the junction the agent needs to decide whether it chooses the left or the right arm, which depends on whether the movement 
        vector at that point points more up or down. There is a reward counter to keep track of how many times the agent chose the correct arm, i.e. the one with the goal.
        TD or STDP learning can be ON or OFF depending on whether the weights of the SR are frozen or not. 
        Besides the different movement policy, the goal counter and the freezing of weights, i.e. use of previously learned SR, this function is similar to runRat().
        Args:
            runTime (int, optional): How long to explore in minutes. Defaults to 10.
            saveEvery (int, optional): Frequency to save snapshots, in minutes. Defaults to 0.5.
            TDSRLearn (bool,optional): toggles whether to do TD learning
            STDPLearn (bool, optional): toggles whether to do STDP learning
            condition (str, optional): which condition to run: 'theta', 'notheta' or 'scrambled'. Defaults to 'theta'. This is for extracting the correct spike data for the movement policy update, NOT the learning condition!
            phase_threshold: determines which spikes count as late phase when computing the population direction vector.
            t_window: determines how far back in time to look for spikes when computing the population direction vector. If None, looks back to the last 2 theta cycles.
            # preTrain (bool, optional): whether to pretrain the agent for 30 minutes before starting the decision task. Defaults to True. Pretraining can help the agent learn a good SR before starting the decision task, which can improve performance.
        NOTES: For this function to run properly, the Maze needs to be a T-maze! Bcs movement policy is only defiend for T-maze.  
        """
        steps = int(runTime * 60 / self.dt) #number of steps to perform
        hist_t = np.zeros(steps)
        hist_pos = np.zeros((steps,2)) # saves the position of the agent at each time step
        hist_delta = np.zeros(steps) # saves the TD learning update size at each time
        reward_counter = 0 # counts how many times the agent reached the goal
        wrong_arm_counter = 0
        self.runDurations = [] # list to store run durations for later analysis
        # save decision start time
        t0 = self.t
        self.decisionStartTime = t0
        # set run start time to later calculate duration of run from start location to goal and compute average time to reach goal
        runStartTime = self.t # to calculate duration of run from start location to goal and compute average time to reach goal

        # add wall to maze to close t-maze stem during task so agent cant escape
        if self.mazeType == 'TMaze':
            rs = self.roomSize
            corridor_half_width = 0.05 * rs # half width of the corridor, to make sure wall is wide enough to block it
            self.walls['CloseStem'] = np.array([[[0.0, rs - corridor_half_width], [0.0, rs + corridor_half_width]]])
            self.mazeState['walls'] = self.walls.copy()
            self.toggleDoors(doorsClosed = False) # make sure right arm is open
            self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords, stateType = self.stateType) # recalculate discrete states to account for new wall
            #create snapshot of maze after adding wall to save the new maze state
            snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta':[self.W_notheta.copy()], 'W_scrambled':[self.W_scrambled.copy()], 'mazeState':[self.mazeState]})
            self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)

        # set initial position of the agent to the stem of the T-maze
        self.pos = np.array([0.2, self.roomSize]) # added to reset position of agent
        self.dir = np.array([1.0, 0.0])
        if self.mazeType == 'TMaze':
            self.LRDecisionPending = True
            self.lastArmChoice = 1

        # cehck whether goal was set
        if (self.goalPos is None) or (self.goalRadius is None) or (self.goalReward is None):
            print("Goal not properly set. Please set goalPos, goalRadius and goalReward to run decision task.")
            return
        
        lastTDstep, distanceToTD = 0, np.random.exponential(self.TDdx) #2cm scale, for determining when to do TD learning step
        
        old_eta = self.eta # save old learning rate for STDP, so we can reset it after the decision task if we are freezing weights
        last_i = 0 # to keep track of how many steps we actually take, in case of keyboard interrupt, so we can save the correct amount of history data
        try: 
        # main task/ training loop and movement policy update
            for i in tqdm(range(steps)): #main training loop
                last_i = i

                if i == 0:
                    hist_pos[i] = self.pos
                    hist_t[i] = self.t
                    hist_delta[i] = 0
                    continue

                try: # try-except block to allow for keyboard interrupt without losing data
                    prevPos = self.pos.copy()
                    # update position, velocity, direction and time according to movement policy
                    self.sweepMovementPolicyUpdate(condition=condition, phase_threshold=phase_threshold, t_window=t_window, maxTurnAngle=maxTurnAngle, maxAngle=maxAngle) # movement policy update for decision task, which uses the recent late phase spikes to determine movement direction
                    if self.enteredGoal(prevPos, self.pos): # check if goal was reached in this step
                        reward_counter += 1
                        runEndTime = self.t
                        runDuration = runEndTime - runStartTime
                        # append run duration to list of run durations for later analysis
                        self.runDurations.append(runDuration)
                        runStartTime = runEndTime # reset run start time for next run
                        print(f"Goal reached! Total times goal reached: {reward_counter}. Duration: {runDuration}. Reset agent to start!")
                        # rest position to starting position after reaching goal and reset direction to face up the stem
                        self.pos = np.array([0.2, np.random.normal(self.roomSize, 0.05)])
                        self.dir = np.array([1.0, 0.0])
                        self.LRDecisionPending = True
                    elif self.pos[1] < 2.0:
                        wrong_arm_counter += 1
                        runEndTime = self.t
                        runStartTime = runEndTime # reset run start time for next run
                        print("Chose wrong arm, resetting position to maze start")
                        self.pos = np.array([0.2, np.random.normal(self.roomSize, 0.05)])
                        self.dir = np.array([1.0, 0.0])
                        self.LRDecisionPending = True

                    if i > 1: # if this is not the first step, we can do learning updates

                        # print(self.pos)
                        """STDP learning step"""
                        if (STDPLearn == True) and (self.stateType in ['bump', 'gaussian', 'gaussianCS', 'gaussianThreshold', 'circles']):
                            if updateSTDPWeights == False:
                                self.eta = 0.0 # if we don't want to update weights/freeze them ste learning rate to 0
                            else:
                                self.eta = old_eta # if weights are not frozen, make sure we use original eta value
                                
                            if self.use_full_STDP_rule == True:
                                _ = self.STDPLearningStep_detailed(dt=self.t - hist_t[i - 1])
                            else:
                                _ = self.STDPLearningStep(dt=self.t - hist_t[i - 1])
                    
                        """TD learning step"""
                        if TDSRLearn == True:
                            alpha = self.alpha
                            try:
                                alpha_ = alpha[0] * np.exp(-(i / steps) * (np.log(self.alpha[0] / self.alpha[1])))  # decaying alpha
                            except:
                                alpha_ = self.alpha

                            if np.linalg.norm(self.pos - hist_pos[lastTDstep]) >= distanceToTD:  # if it's moved over 2cm meters from last step
                                dtTD = self.t - hist_t[lastTDstep]
                                delta = self.TDLearningStep(pos=self.pos, prevPos=hist_pos[lastTDstep], dt=dtTD, tau=self.tau, alpha=alpha_)
                                lastTDstep = i
                                distanceToTD = np.random.exponential(self.TDdx)
                                hist_delta[i] = delta
                
                    # update clean theta phase but not scrambled theta phase
                    self.thetaPhase = self.thetaFreq * (self.t % (1 / self.thetaFreq)) * 2 * np.pi # 10 Hz by default

                    # update history arrays
                    hist_pos[i] = self.pos
                    hist_t[i] = self.t

                    # save snapshot
                    if (isinstance(saveEvery, numbers.Number)) and (i % int(saveEvery * 60 / self.dt) == 0):
                        snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta': [self.W_notheta.copy()], 'W_scrambled': [self.W_scrambled.copy()], 'mazeState': [self.mazeState]})  
                        self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)
            
                except KeyboardInterrupt:
                    print("Keyboard Interrupt:")
                    break

            self.runID += 1
            runHistory = pd.DataFrame({'t': list(hist_t[:last_i + 1]), 'pos': list(hist_pos[:last_i + 1]), 'delta': list(hist_delta[:last_i + 1])})
            self.history = pd.concat([self.history, runHistory], ignore_index=True)
            snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta': [self.W_notheta.copy()], 'W_scrambled': [self.W_scrambled.copy()], 'mazeState': [self.mazeState]})  
            self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)

            #find and save grid/place cells
            print("Calculating place and grid cells")
            self.gridFields = self.getGridFields(self.M)
            self.placeFields = self.getPlaceFields(self.M)

            if TDSRLearn == True: # 
                plotter = Visualiser(self)
                plotter.plotTrajectory(starttime=(self.t / 60) - 0.2, endtime=self.t / 60)
                delta = np.array(hist_delta) # extract TD learning update sizes from history
                time = np.array(hist_t) # extract time points from history
                time = time[delta != 0] / 60 # only keep time points where there was a TD update, and convert to minutes
                delta = delta[delta != 0] # only keep TD update sizes that are non-zero
                time, delta = time[::10], delta[::10] # downsample for smoother plotting
                smooth_delta = [np.mean(delta[max(0, i - 100):min(i + 100, len(delta))]) for i in range(len(delta))] # compute a smoothed version of the TD update sizes using a moving average
                fig, ax = plt.subplots(figsize=(2, 1))
                ax.scatter(time, delta, s=0.5, alpha=0.5)
                ax.scatter(time, smooth_delta, s=1, alpha=0.5, c='C2')
                ax.set_xlabel("Time / min")
                ax.set_ylabel("Update size")
            
            # performance metrics/ summary
            print(f"Performance Metrics:")
            print(f"  - Reward count: {reward_counter}")
            print(f"  - Total reward: {reward_counter * self.goalReward}")
            if reward_counter or wrong_arm_counter >0:
                print(f"  - Percentage of correct choices: {100 * (reward_counter / (reward_counter + wrong_arm_counter)):.2f}%")
                print(f"  - average time to reach goal: {np.mean(self.runDurations):.2f} seconds")
            print("================================")
            # plot maze structure to make sure wall to close stem is actually at the right spot (before removing it again at end of function)
            #plotter = Visualiser(self)
            #plotter.plotMazeStructure() # -> wall indeed at right spot!

            #return number of time reward was reached and total reward obtained
            return t0, {'reward_count': reward_counter, 'total_reward': reward_counter * self.goalReward}
        finally:
            if 'CloseStem' in getattr(self, 'walls', {}):
                del self.walls['CloseStem'] # remove wall closing stem after decision task is done, so agent can explore whole maze again
                self.mazeState['walls'] = self.walls.copy()
                self.toggleDoors(doorsClosed = False) # make sure right arm is open again
                self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords, stateType=self.stateType) # recalculate discrete states to account for removed wall 
            self.eta = old_eta # reset learning rate for STDP to original value after decision task is done, in case it was changed to freeze weights

    def TaskEpisode(self,
                    saveEvery=0.5, 
                    TDSRLearn=True, 
                    STDPLearn=True, 
                    condition = 'theta', 
                    phase_threshold = 1.5*np.pi, 
                    t_window =None,
                    updateSTDPWeights = False,
                    maxTurnAngle = False,
                    maxAngle = None,
                    timeout = 90.0):
        """
        Runs a signle episode of the decision task: the agent starts at the starting position and navigates the maze 
        until it finds the reward, selects the wrong arm or reaches a time limit. The episode can be repeated multiple times 
        by calling this function multiple times, and the performance metrics will be saved for each episode. 
        """
        if timeout is not None:
            timeout = timeout
        else:
            print("No timeout set! Please set a timeout to prevent infinite episodes!")
        reward_counter = 0
        wrong_arm_counter = 0
        timeout_counter = 0
        episodeDuration = 0.0
        episodeStartTime = self.t
        steps = int(timeout/self.dt) # max number of steps until timeout
        hist_t = np.zeros(steps) # save time at each step
        hist_pos = np.zeros((steps,2)) # save pos of agent at each time step dt
        hist_delta = np.zeros(steps) # saves TD learning update size

        # set initial position and direction of agent at start of episode at stem of T-maze
        self.pos = np.array([0.2, np.random.normal(self.roomSize, 0.05)])
        self.dir = np.array([1.0, 0.0])
        if self.mazeType == 'TMaze':
            self.LRDecisionPending = True
            self.lastArmChoice = 1
        
        # check whether goal was set
        if (self.goalPos is None) or (self.goalRadius is None) or (self.goalReward is None):
            print("Goal not properly set. Please set goalPos, goalRadius and goalReward to run task episode.")
            return
        lastTDstep, distanceToTD = 0, np.random.exponential(self.TDdx) #2cm scale, for determining when to do TD learning step
        
        old_eta = self.eta # save old learning rate for STDP, so we can reset it after the episode if we are freezing weights
        last_i = 0 # to keep track of how many steps we actually take, in case of keyboard interrupt, so we can save the correct amount of history data
        try:
            for i in tqdm(range(steps)):
                last_i = i

                if i == 0:
                    hist_pos[i] = self.pos
                    hist_t[i] = self.t
                    hist_delta[i] = 0
                    continue

                try:
                    prevPos = self.pos.copy()
                    # update position, velocity, direction and time according to movement policy
                    self.sweepMovementPolicyUpdate(condition=condition, phase_threshold=phase_threshold, t_window=t_window, maxTurnAngle=maxTurnAngle, maxAngle=maxAngle) # movement policy update for decision task, which uses the recent late phase spikes to determine movement direction
                    if self.enteredGoal(prevPos, self.pos):
                        reward_counter +=1
                        runEndTime = self.t
                        episodeDuration = runEndTime - episodeStartTime
                        print(f"Goal reached! Duration: {episodeDuration}. Starting new episode!")
                        break # end episode when goal is reached
                    elif self.pos[1] < 2.0:
                        wrong_arm_counter +=1
                        runEndTime = self.t
                        episodeDuration = runEndTime - episodeStartTime
                        print(f"Wrong arm chosen! Duration: {episodeDuration}. Starting new episode!")
                        break # end episode when wrong arm is chosen
                    elif self.t - episodeStartTime >= (timeout - 5.0): # 1e-9 to account for rounding errors
                    #elif i >= steps - 1: # check if reached last step before timeout
                        print(f"Episode timed out after {timeout} seconds! Starting new episode!")
                        timeout_counter +=1
                        runEndTime = self.t
                        episodeDuration = runEndTime - episodeStartTime
                        #print(f"Episode timed out after {timeout} seconds! Starting new episode!")
                        break # end episode when timeout is reached

                    if i > 1: # if this is not the first step, we can do learning updates
                        """STDP learning step"""
                        if (STDPLearn == True) and (self.stateType in ['bump', 'gaussian', 'gaussianCS', 'gaussianThreshold', 'circles']):
                            if updateSTDPWeights == False:
                                self.eta = 0.0
                            else: 
                                self.eta = old_eta

                            if self.use_full_STDP_rule == True:
                                _ = self.STDPLearningStep_detailed(dt=self.t - hist_t[i-1])
                            else:
                                _ = self.STDPLearningStep(dt=self.t - hist_t[i-1])
                        
                        """TD learning step"""
                        if TDSRLearn == True:
                            alpha = self.alpha
                            try:
                                alpha_ = alpha[0] * np.exp(-(i / steps) * (np.log(self.alpha[0] / self.alpha[1])))  # decaying alpha
                            except:
                                alpha_ = self.alpha

                            if np.linalg.norm(self.pos - hist_pos[lastTDstep]) >= np.random.exponential(self.TDdx):  # if it's moved over 2cm meters from last step
                                dtTD = self.t - hist_t[lastTDstep]
                                delta = self.TDLearningStep(pos=self.pos, prevPos=hist_pos[lastTDstep], dt=dtTD, tau=self.tau, alpha=alpha_)
                                lastTDstep = i
                                distanceToTD = np.random.exponential(self.TDdx)
                                hist_delta[i] = delta

                    # update clean theta phase
                    self.thetaPhase = self.thetaFreq * (self.t % (1 / self.thetaFreq)) * 2 * np.pi # 10 Hz by default

                    # update history arrays
                    hist_pos[i] = self.pos
                    hist_t[i] = self.t

                    # save snapshot
                    if (isinstance(saveEvery, numbers.Number)) and (i % int(saveEvery * 60 / self.dt) == 0):
                        snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta': [self.W_notheta.copy()], 'W_scrambled': [self.W_scrambled.copy()], 'mazeState': [self.mazeState]})  
                        self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)

                except KeyboardInterrupt:
                    print("Keyboard Interrupt:")
                    break

            # after episode is done, save history and snapshots, and calculate place and grid fields
            self.runID += 1
            runHistory = pd.DataFrame({'t': list(hist_t[:last_i + 1]), 'pos': list(hist_pos[:last_i + 1]), 'delta': list(hist_delta[:last_i + 1])})
            self.history = pd.concat([self.history, runHistory], ignore_index=True)
            snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'W_notheta': [self.W_notheta.copy()], 'W_scrambled': [self.W_scrambled.copy()], 'mazeState': [self.mazeState]})  
            self.snapshots = pd.concat([self.snapshots, snapshot], ignore_index=True)
            
            print("Calculating place and grid cells")
            self.gridFields = self.getGridFields(self.M)
            self.placeFields = self.getPlaceFields(self.M)

            return {'reward_count': reward_counter, 'wrong_arm_count': wrong_arm_counter, 'timeout_count': timeout_counter, 'episode_duration': episodeDuration}
        finally:
            self.eta = old_eta # reset learning rate for STDP to original value after episode is done, in case it was changed to freeze weights



    def TDLearningStep(self, pos, prevPos, dt, tau, alpha):
        """TD learning step
            Improves estimate of SR matrix, M, by a TD learning step. 
            By default this is done using learning rule for generic feature vectors (see de Cothi and Barry 2020). 
            If stateType is onehot, additional efficiencies can be gained by using onehot specific learning rule (see Stachenfeld et al. 2017)
            Does time continuous TD learning (see Doya, 2000)
        Args:
            pos: position at t+dt (t)
            prevPos (array): position at t (t-dt)
            dt (float): time difference between two positions
            tau (float or int): memory decay time (analogous to gamma in TD, gamma = 1 - dt/tau)
            alpha (float): learning rate
            mask (bool or str): whether to mask TM update to update only cells near current location
            asynchronus (bool): update cells asynchronusly (like hopfield)
        """
        state = self.posToState(pos,stateType=self.stateType) 
        prevState = self.posToState(prevPos,stateType=self.stateType) 

        data = ( (state,                        prevState,                    self.M        ) ,
                 (self.thetaModulation(state),  self.thetaModulation(state),  self.M_theta) )

        
        for i, (state, prevState, M) in  enumerate(data): 
            #onehot optimised TD learning 
            if self.stateType == 'onehot': 
                s_t = np.argwhere(prevState)[0][0]
                s_tplus1 = np.argwhere(state)[0][0]
                Delta = state + (tau / dt) * ((1 - dt/tau) * M[:,s_tplus1] - M[:,s_t])
                M[:,s_t] += alpha * Delta - 2 * alpha * self.TDreg * M[:,s_t]

            #normal TD learning 
            else:
                delta = ((tau * dt) / (tau + dt)) * self.successorFeatureNorm * prevState + M @ ((tau/(tau + dt))*state - prevState)
                Delta = np.outer(delta, prevState)
                M += alpha * Delta - 2 * alpha * self.TDreg * M #regularisation
            
            if i == 0: 
                Del = Delta

        return np.linalg.norm(Del)

    

    def STDPLearningStep(self,dt):       
        """Takes the curent theta phase and estimate firing rates for all basis cells according to a simple theta sweep model. 
           From here it samples spikes and performs STDP learning on a weight matrix.

        Args:
            dt (float): Time step length 

        Returns:
            float array: vector of firing rates for this time step 
        """   
        state = self.posToState(self.pos)

        data = ( (state,
                self.W_notheta,  
                self.preTrace_notheta,  
                self.postTrace_notheta,  
                self.lastSpikeTime_notheta, 
                self.spikeCount_notheta),

                 (self.thetaModulation(state),
                self.W,          
                self.preTrace,          
                self.postTrace,          
                self.lastSpikeTime, 
                self.spikeCount), 
                
                (self.thetaModulation_scrambled(state),
                 self.W_scrambled,
                 self.preTrace_scrambled,
                 self.postTrace_scrambled,
                 self.lastSpikeTime_scrambled,
                 self.spikeCount_scrambled), 
                )

        
        for i, (firingRate, W, preTrace, postTrace, lastSpikeTime, spikeCount) in enumerate(data): 
            firingRate_ = self.peakFiringRate * firingRate + self.baselineFiringRate #scale firing rate and add noise
            lam = self._sanitize_poisson_lambda(firingRate_ * dt, label="STDPLearningStep")
            n_spike_list = np.random.poisson(lam)
            
            spikingNeurons = (n_spike_list != 0) #in short time dt cells can spike 0 or 1 time only (good enough approximation) 
            spikeCount += sum(spikingNeurons)
            spikeTimes = np.random.uniform(self.t,self.t+dt,self.nCells)[spikingNeurons]
            spikeIDs = np.arange(self.nCells)[spikingNeurons]
            spikeList = np.vstack((spikeIDs,spikeTimes)).T
            spikeList = spikeList[np.argsort(spikeList[:,1])]   

            for spikeInfo in spikeList:
                cell, time = int(spikeInfo[0]), spikeInfo[1] 
                timeDiff = time - lastSpikeTime 


                preTrace        *= np.exp(- timeDiff / self.tau_STDP_plus) #traces for all cells decay...
                postTrace       *= np.exp(- timeDiff / self.tau_STDP_minus) #traces for all cells decay...
                W[cell,:]       += self.eta * preTrace #weights to postsynaptic neuron (should increase when post fires)
                W[:,cell]       += self.eta * postTrace #weights to presynaptic neuron (should decrease when post fires) 
                postTrace[cell] += self.a_STDP  #update trace (post trace probably negative)
                preTrace[cell]  += 1 #update trace 



                lastSpikeTime += timeDiff

            if i == 1: 
                thetaFiringRate = firingRate_
            if i == 2:
                scrambledThetaFiringRate = firingRate_

        return thetaFiringRate

    def STDPLearningStep_detailed(self,dt):       
        """Takes the curent theta phase and estimate firing rates for all basis cells according to a simple theta sweep model. 
           From here it samples spikes and performs STDP learning on a weight matrix.

        Args:
            dt (float): Time step length 

        Returns:
            float array: vector of firing rates for this time step 
        """   
        state = self.posToState(self.pos)

        data = ( (state,
                self.W_notheta,  
                self.preTrace_notheta,  
                self.postTrace_notheta,  
                self.lastSpikeTime_notheta, 
                self.spikeCount_notheta),

                 (self.thetaModulation(state),
                self.W,          
                self.preTrace,          
                self.postTrace,          
                self.lastSpikeTime, 
                self.spikeCount), 
                
                (self.thetaModulation_scrambled(state),
                 self.W_scrambled,
                 self.preTrace_scrambled,
                 self.postTrace_scrambled,
                 self.lastSpikeTime_scrambled,
                 self.spikeCount_scrambled),
                )
        # fixed order of conditions in 'data'
        condition_names = ('notheta', 'theta', 'scrambled')

        # safety block for old objects loaded from disk that might not have the new 'spikedata_by_condition' attribute
        if not hasattr(self, "spikedata_by_condition"):
            self.spikedata_by_condition = {
                'notheta':   {'CA1': {'times': [], 'ids': []}, 'CA3': {'times': [], 'ids': []}},
                'theta':     {'CA1': {'times': [], 'ids': []}, 'CA3': {'times': [], 'ids': []}},
                'scrambled': {'CA1': {'times': [], 'ids': []}, 'CA3': {'times': [], 'ids': []}},
            }
        
        for i, (firingRate, W, preTrace, postTrace, lastSpikeTime, spikeCount) in enumerate(data): 
            preFiringRate_ = self.peakFiringRate * firingRate + self.baselineFiringRate #scale firing rate and add noise
            if self.online_mapping == "identity":
                mapMatrix =  np.identity(self.nCells)
            elif self.online_mapping == "Widentity":
                mapMatrix =  W + 0.5*np.identity(self.nCells)
            elif self.online_mapping == "W":
                mapMatrix =  W
            else: 
                mapMatrix = self.online_mapping

            postFiringRate_ = np.maximum(0,np.matmul(mapMatrix,preFiringRate_)) # multiply pre activity by synaptic weight matrix -> input to post neurons. Also ensure all firing rates are above 0
            firingRate_ = np.concatenate((preFiringRate_,postFiringRate_)) # concatenate pre and post firing rates in one long vector -> length = 2xnCells
            layerLabel_ = np.array(['pre']*len(preFiringRate_) + ['post']*len(postFiringRate_)) # label which neuron belongs to which layer -> labeled 'pre' or 'post'
            neuronIDs = np.concatenate((np.arange(len(preFiringRate_)), np.arange(len(postFiringRate_)))) # create neuron ID -> e.g neuron 10 in pre layer and neuron 10 in post layer (they are different neurons distinguished by pre/post label)
            lam = self._sanitize_poisson_lambda(firingRate_ * dt, label="STDPLearningStep_detailed")
            n_spike_list = np.random.poisson(lam) # get the firing rate from random poisson distribution
            # n_spike_list[i] would be the number of spikes of neuron i in the timestep

            spikingNeurons = (n_spike_list != 0) #in short time dt cells can spike 0 or 1 time only (good enough approximation) 
            spikeCount += sum(spikingNeurons) # counts how many neurons spiked in dt and gives sum
            spikeTimes = np.random.uniform(self.t,self.t+dt,len(neuronIDs))[spikingNeurons]
            spikeIDs = neuronIDs[spikingNeurons] # get ID of the neurons that spiked
            spikeLayerLabels = layerLabel_[spikingNeurons] # get the layer they belonged to

            # numeric sorting by time
            order = np.argsort(spikeTimes)
            spikeTimes = spikeTimes[order]
            spikeIDs = spikeIDs[order]
            spikeLayerLabels = spikeLayerLabels[order]
            # store in spikeList
            spikeList = np.vstack((spikeIDs,spikeTimes,spikeLayerLabels)).T # put them all in a matrix/list together
            # spikeList = spikeList[np.argsort(spikeList[:,1])] # sort them according to spike time   #ed by Livi to use ordering in '#numeric sorting by time' 

            # create a new block to save the CA1 spikes of the specific theta condition (Livi addition)
            cond = condition_names[i]
            ca1_mask = (spikeLayerLabels == 'post')
            ca3_mask = (spikeLayerLabels == 'pre')
            self.spikedata_by_condition[cond]['CA3']['times'].extend(spikeTimes[ca3_mask].tolist())
            self.spikedata_by_condition[cond]['CA3']['ids'].extend(spikeIDs[ca3_mask].tolist())
            self.spikedata_by_condition[cond]['CA1']['times'].extend(spikeTimes[ca1_mask].tolist())
            self.spikedata_by_condition[cond]['CA1']['ids'].extend(spikeIDs[ca1_mask].tolist())

            # still save in spikedata as well for backward compatibility 
            self.spikedata['CA3']['times'].extend(spikeTimes[ca3_mask].tolist())
            self.spikedata['CA3']['ids'].extend(spikeIDs[ca3_mask].tolist())
            self.spikedata['CA1']['times'].extend(spikeTimes[ca1_mask].tolist())
            self.spikedata['CA1']['ids'].extend(spikeIDs[ca1_mask].tolist())


            for spikeInfo in spikeList:
                cell, time, layer = int(spikeInfo[0]), float(spikeInfo[1]), spikeInfo[2] #for each column cell = cell ID as integer, time= spike time as float, layer= string I guess
                timeDiff = time - lastSpikeTime # difference between current time and the last processed spike event of any neuron (not that specific neuron!)

                preTrace        *= np.exp(- timeDiff / self.tau_STDP_plus) #traces for all cells decay...
                postTrace       *= np.exp(- timeDiff / self.tau_STDP_minus) #traces for all cells decay...
                if layer == 'pre':
                    W[:,cell]       += self.eta * postTrace #weights from presynaptic neuron should decrease when pre fires (post-before-PRE) 
                    preTrace[cell]  += 1 #update trace 
                if layer == 'post':
                    W[cell,:]       += self.eta * preTrace #weights to postsynaptic neuron should increase when post fires (pre-before-POST)
                    postTrace[cell] += self.a_STDP  #update trace (post trace probably negative)

                lastSpikeTime += timeDiff

            if i == 1: # if theta condition (1=theta; 0=notheta, 2 from line 538)
                thetaFiringRate = firingRate_ #store theta firing rate to return later
            if i == 2: # if scrambled theta condition (2=scrambled, from line 538)
                scrambledThetaFiringRate = firingRate_ # store scrambled theta firing rate in case it is needed at any point in the future (Livi addition)
        
        if self.rownorm == True: 
            # self.W = self.W / np.linalg.norm(self.W,axis=1)[:,np.newaxis]
            # self.W_notheta = self.W_notheta / np.linalg.norm(self.W_notheta,axis=1)[:,np.newaxis]
            sumW = np.sum(self.W,axis=1)
            sumW[sumW<1]=1
            self.W = self.W / sumW[:,np.newaxis]
            sumWnt = np.sum(self.W_notheta,axis=1)
            sumWnt[sumWnt<1]=1
            self.W_notheta = self.W_notheta / sumWnt[:,np.newaxis]
            # add part about scrambled theta condition (Livi addition)
            sumWscrambled = np.sum(self.W_scrambled,axis=1)
            sumWscrambled[sumWscrambled<1]=1
            self.W_scrambled = self.W_scrambled / sumWscrambled[:,np.newaxis]
        
        #save spike data #ed bcs added saving separately per condition 
        # CA3spiketimes = spikeTimes[spikeLayerLabels=='pre'] # spike time of all CA3 layer neurons
        #CA3spikeids = spikeIDs[spikeLayerLabels=='pre'] # spike IDs of CA3 cells
        #CA1spiketimes = spikeTimes[spikeLayerLabels=='post'] # spike times of CA1 layer cells
        #CA1spikeids = spikeIDs[spikeLayerLabels=='post'] # spike IDs of CA1 layer cells
        #self.spikedata['CA3']['times'].extend(CA3spiketimes)
        #self.spikedata['CA3']['ids'].extend(CA3spikeids)
        #self.spikedata['CA1']['times'].extend(CA1spiketimes)
        #self.spikedata['CA1']['ids'].extend(CA1spikeids)

        return thetaFiringRate

    def _sanitize_poisson_lambda(self, lam, label="STDP"):
        """Make Poisson lambda numerically safe (finite, non-negative, bounded)."""
        lam_arr = np.asarray(lam, dtype=float)

        # Poisson requires finite and non-negative lambda values.
        finite_nonneg = np.isfinite(lam_arr) & (lam_arr >= 0)
        if not np.all(finite_nonneg):
            if not hasattr(self, "_poisson_invalid_warned"):
                self._poisson_invalid_warned = False
            if self._poisson_invalid_warned is False:
                print(f"{label}: replacing non-finite/negative Poisson lambda values with 0.")
                self._poisson_invalid_warned = True
            lam_arr = np.where(finite_nonneg, lam_arr, 0.0)

        # Guard against extreme values that can overflow numpy's Poisson sampler.
        max_lam = 1e6
        too_large = lam_arr > max_lam
        if np.any(too_large):
            if not hasattr(self, "_poisson_clip_warned"):
                self._poisson_clip_warned = False
            if self._poisson_clip_warned is False:
                print(f"{label}: clipping large Poisson lambda values to {max_lam:g}.")
                self._poisson_clip_warned = True
            lam_arr = np.minimum(lam_arr, max_lam)

        return lam_arr

    
    def thetaModulation(self, firingRate, position=None, direction=None):
        """Takes a firing rate vector and modulates it to account for theta phase precession

        Args:
            firingRate (np.array): The raw (position dependent) firing rate vector to be modulated 
            position (np.array(2,), optional): The agent position. Defaults to None.
            direction (np.array(2,), optional): The agent direction. Defaults to None.
        """        
        if position is None:
            position = self.pos
        if direction is None:
            direction = self.dir

        vectorToCells = self.vectorsToCellCentres(position)
        sigmasToCellMidline = (np.dot(vectorToCells,direction) / np.linalg.norm(direction))  / self.sigmas #as mutiple of sigma
        preferedThetaPhase = np.pi + sigmasToCellMidline * self.precessFraction * np.pi

        phaseDiff = preferedThetaPhase - self.thetaPhase
        modulatedFiringRate = firingRate * vonmises.pdf(phaseDiff,kappa=self.kappa) * 2*np.pi

        return modulatedFiringRate
    
    # Add a new function that implements scrambled theta modulation (Livi addition)
    def thetaModulation_scrambled(self, firingRate, position=None, direction=None):
        """
        REWRITE(THIS IS NOT CORRECT): Takes a firing rate vector and modulates it to account for scrambled theta phase precession, 
        i.e. each cell has a random preferred theta phase that does not depend on position.
        Args:
            firingRate (np.array): The raw (position dependent) firing rate vector to be modulated
            position (np.array(2,), optional): The agent position. Defaults to None.
            direction (np.array(2,), optional): The agent direction. Defaults to None.
            """
        if position is None:
            position = self.pos
        if direction is None:
            direction = self.dir
        
        drift = self.thetaFreq * self.dt * 2*np.pi # calculate how much the theta phase should drift in this time step
        scramble_noise = np.random.normal(0, self.scramble_strength) # adds some noise drawn from a normal distribution with mean 0 and sd defined by scramble_strength parameter to the scrambled theta phase at each time step, making it more random and less predictable
        hf_jitter = np.random.normal(0, self.hf_strength) # adds some high frequency jitter noise 
        self.thetaPhase_scrambled = (self.thetaPhase_scrambled + drift + scramble_noise + hf_jitter) % (2*np.pi) # update scrambled theta phase by adding drift, scramble noise and high frequency jitter, and wrap around to keep it between 0 and 2*pi for vonmises calculation

        # save the scrambled theta phase for this time step (Livi addition)
        self.thetaPhase_scrambled_history.append(float(np.asarray(self.thetaPhase_scrambled).reshape(-1)[0]))
        if not hasattr(self, 'thetaPhase_scrambled_time_history'):
            self.thetaPhase_scrambled_time_history = []
        self.thetaPhase_scrambled_time_history.append(float(self.t))

        ## OLD SCRAMBLE VERSION
        # phase_inc = 2*np.pi * self.thetaFreq * self.dt
        # phase_noise = np.random.normal(0, np.pi/3, size=self.nCells) # add some noise to the scrambled phase to make it less extreme
        # self.thetaPhase_scrambled = (self.thetaPhase_scrambled + phase_inc + phase_noise) % (2*np.pi) # update scrambled theta phase with some noise

        ## same as for normal theta modulation, just with scrambled theta phase instead of position-dependent preferred phase
        vectorToCells = self.vectorsToCellCentres(position)
        sigmasToCellMidline = (np.dot(vectorToCells,direction) / np.linalg.norm(direction))  / self.sigmas #as mutiple of sigma
        preferedThetaPhase = np.pi + sigmasToCellMidline * self.precessFraction * np.pi

        phaseDiff = preferedThetaPhase - self.thetaPhase_scrambled
        modulatedFiringRate = firingRate * vonmises.pdf(phaseDiff,kappa=self.kappa) * 2*np.pi

        return modulatedFiringRate
        

    def movementPolicyUpdate(self):
        """Movement policy update. 
            In principle this does a very simple thing: 
            • updates time by dt, 
            • updates position along the velocity direction 
            • updates velocity (speed and direction) accoridng to a movement policy
            In reality it's a complex function as the policy requires checking for immediate or upcoming collisions with all walls at each step.
            This is done by function self.checkWallIntercepts()
            What it does with this info (bounce off wall, turn to follow wall, etc.) depends on policy. 
        """

        dt = self.dt
        self.t += dt
        proposedNewPos = self.pos + self.speed * self.dir * dt
        proposedStep = np.array([self.pos,proposedNewPos])

        if (self.biasDoorCross == True) and (self.mazeType == 'twoRooms'): 
            #if agent crosses into door zone there's its turn direction is biased to try and cross the door 
            #this is done by setting agents direction in the right direction and not changing it again until after it's crossed
            doorRegionSize = 1
            if self.doorPassage == False:
                #if step cross into door region
                if (np.linalg.norm(self.pos - np.array([self.roomSize,self.roomSize/2])) > doorRegionSize) and (np.linalg.norm(proposedNewPos - np.array([self.roomSize,self.roomSize/2])) < doorRegionSize) and (abs(self.pos[0] - self.roomSize) > 0.01):
                    if 100*np.random.uniform(0,1) < 50: #start a doorPassage 
                        self.doorPassage = True
                        self.doorPassageTime = self.t
                        return
                    else: #ignore this 
                        pass
            if self.doorPassage == True: 
                if ((self.pos[0]<(self.roomSize)) != (proposedNewPos[0]<(self.roomSize))) or ((self.t - self.doorPassageTime)*self.speedScale > 2*doorRegionSize):
                    self.doorPassage = False
                    if ((self.pos[0]<(self.roomSize)) != (proposedNewPos[0]<(self.roomSize))): 
                        print("crossed",self.t)
                    if ((self.t - self.doorPassageTime)*self.speedScale > 2*doorRegionSize):
                        print("time")
        
        if (self.biasToGoal == True) and (self.mazeType == 'oneRoom'):
            #if agent is within a certain distance of the goal, its turn direction is biased to try and go towards the goal 
            #this is done by setting agents direction in the right direction and not changing it again until after it's reached the goal 
            goalRegionSize = 0.8
            if (not hasattr(self, "goalPos")) or (self.goalPos is None):
                self.reachedGoal = False
            elif self.reachedGoal == False:
                #if step cross into goal region
                if (np.linalg.norm(self.pos - self.goalPos) > goalRegionSize) and (np.linalg.norm(proposedNewPos - self.goalPos) < goalRegionSize):
                    if 100*np.random.uniform(0,1) < 70: #start a goal bias
                        self.reachedGoal = True
                        self.goalReachedTime = self.t
                    else: #ignore this
                        pass
            if self.reachedGoal == True:
                if (np.linalg.norm(self.pos - self.goalPos) < goalRegionSize) and (np.linalg.norm(proposedNewPos - self.goalPos) > goalRegionSize):
                    self.reachedGoal = False
                    print("left goal region",self.t)
                if (self.t - self.goalReachedTime)*self.speedScale > 2*goalRegionSize:
                    self.reachedGoal = False
                    print("time",self.t)

        
        checkResult = self.checkWallIntercepts(proposedStep)
        if self.movementPolicy == 'randomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                randomTurnSpeed = np.random.normal(0,self.rotSpeedScale)
                self.dir = turn(self.dir,turnAngle=randomTurnSpeed*dt)
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'trueRandomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                self.dir = turn(self.dir,turnAngle=np.random.uniform(0,2*np.pi))
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'leftRightRandomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                self.dir = turn(self.dir,turnAngle=np.random.choice([0,np.pi]))
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'raudies':
            if checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
            elif ((checkResult[0] == 'collisionAhead') and (self.biasWallFollow==True)):
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'follow')
            elif (checkResult[0] == 'noImmediateCollision') or (((checkResult[0] == 'collisionAhead') and (self.biasWallFollow==False))):
                self.pos = proposedNewPos

            
            self.speed = np.random.rayleigh(self.speedScale)
            if self.t - self.lastTurnUpdate >= 0.1: #turn updating done at intervals independednt of dt or else many small turns cancel out but few big ones dont 
                randTurnMean = 0
                if self.doorPassage == True: 
                        d_theta  = theta(self.dir) - theta(np.array([self.roomSize,self.roomSize/2]) - self.pos)
                        if d_theta > 0: randTurnMean = -self.rotSpeedScale
                        else: randTurnMean = self.rotSpeedScale
                if self.reachedGoal == True and hasattr(self, "goalPos") and (self.goalPos is not None):
                        d_theta_goal = theta(self.dir) - theta(np.array(self.goalPos) - self.pos)
                        if d_theta_goal > 0:
                            randTurnMean = -self.rotSpeedScale
                        else:
                            randTurnMean = self.rotSpeedScale
                self.randomTurnSpeed = np.random.normal(randTurnMean,self.rotSpeedScale)
                self.lastTurnUpdate = self.t
            self.dir = turn(self.dir, turnAngle=self.randomTurnSpeed*dt)

        if self.movementPolicy == 'windowsScreensaver':
            if checkResult[0] != 'collisionNow': # [0] vs [1]: checkResult[0] is the type of collision (collisionNow, collisionAhead, noImmediateCollision), checkResult[1] is the wall involved in the collision if there is one
                self.pos = proposedNewPos
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == '1DOrnUhl':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
            self.speed += ornstein_uhlenbeck(dt=dt, x=self.speed, drift=self.speedScale,noise_scale=self.speedScale, coherence_time=5)
            self.speed = max(0,self.speed)

        
        if self.mazeType == 'loop':
            self.pos[0] = self.pos[0] % self.roomSize # if in loop maze, wrap x position around so that when agent goes beyond one end it comes back in the other end
        
        # OLD T-Maze Policy
        #if self.mazeType == 'TMaze':
            #if (self.pos[0] > self.roomSize+0.05) and (self.LRDecisionPending==True): # self.pos[0] is the x position and self.roomSize + 0.5 is the x position/threshold just beyond the junction. If decision pending == True decision still needs to be made
                #if np.random.choice([0,1],p=[0.90,0.10]) == 0: # 66% chance to choose 0 (go left) and 34% chance to choose 1 (go right)
                    #self.dir = np.array([0,1])
                #else:
                    #self.dir = np.array([0,-1])
                #self.LRDecisionPending=False # mark that decision has been made
            #if self.pos[1] > self.extent[3] or self.pos[1] < self.extent[2]:  # check if agent has left the maze in the y direction (i.e. gone down one of the arms of the T maze). self.pos[1] is the y position, self.extent[2] and self.extent[3] are the y boundaries of the maze.  
                #self.pos = np.array([0,1]) # if agent has exited maze in y direction reset its position to the start of the T maze (x=0, y=1)
                #self.dir = np.array([1,0]) # reset its direction to be facing towards the junction (positive x direction)
                #self.LRDecisionPending=True # allow decision to be mae again at next junction
        
        # NEW T-Maze Policy: arms loop back into stem rather than agent being teleported back to start. This is more realistic and allows for overshooting junction and going a bit into the arm before turning, which is common in real rats. It also allows for more naturalistic trajectories where the rat can go back and forth between arms and stem multiple times.
        if self.mazeType == "TMaze":
            corridor_half_width = 0.05 * self.roomSize
            junction_x = self.roomSize + corridor_half_width
            stem_y = self.roomSize
            lane_offset = 0.03 * self.roomSize  # slight separation for re-entry lanes on the stem

            # Decision at junction (once per lap)
            if (self.pos[0] > junction_x) and (self.LRDecisionPending == True):
                if np.random.choice([0, 1], p=[0.90, 0.10]) == 0:
                    # upper arm
                    self.dir = np.array([0, 1])
                    self.lastArmChoice = 1
                else:
                    # lower arm
                    self.dir = np.array([0, -1])
                    self.lastArmChoice = -1
                self.LRDecisionPending = False

            # Closed-loop wrap from arm end back to stem start (preserve overshoot)
            if self.pos[1] > self.extent[3]:
                # exited top arm
                overshoot = self.pos[1] - self.extent[3]
                self.pos = np.array([overshoot, stem_y + lane_offset])
                self.dir = np.array([1, 0])  # continue rightward along stem
                self.LRDecisionPending = True

            elif self.pos[1] < self.extent[2]:
                # exited bottom arm
                overshoot = self.extent[2] - self.pos[1]
                self.pos = np.array([overshoot, stem_y - lane_offset])
                self.dir = np.array([1, 0])  # continue rightward along stem
                self.LRDecisionPending = True

        #catchall instances a rat escapes the maze by accident, pops it 2cm within maze 
        if ((self.pos[0] < self.extent[0]) or 
            (self.pos[0] > self.extent[1]) or 
            (self.pos[1] < self.extent[2]) or 
            (self.pos[1] > self.extent[3])):
            print(self.pos)
            self.pos[0] = max(self.pos[0],self.extent[0]+0.02)
            self.pos[0] = min(self.pos[0],self.extent[1]-0.02)
            self.pos[1] = max(self.pos[1],self.extent[2]+0.02)
            self.pos[1] = min(self.pos[1],self.extent[3]-0.02)
            print("Rat escaped!")
            if self.mazeType == 'TMaze':
                self.dir=np.array([1,0])
                self.LRDecisionPending = True
            # plotter = Visualiser(self)
            # plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)

    def sweepMovementPolicyUpdate(self, condition = None, phase_threshold = None, t_window = None, return_counts = True, maxTurnAngle = False, maxAngle = None):
        """
        A movement policy update function where the movement is determined by the direction of the population vector 
        from the current position of the agent to the cell centres of the late-phase spikes in the most recent theta cycles.
        That means at each time step, the agent looks at which cells have spiked in the last few theta cycles, calculates 
        the population vector from its current position to the centres of those cells, and then moves in that direction.
        Args:
            maxTurnAngle (bool): if True, limit the maximum turn angle of the agent at each step to maxAngle (in degrees), if you want to prevent large turns. If False, no limit on turn angle.
            maxAngle (float, optional): maximum turn angle in degrees if maxTurnAngle is True. Defaults to None.
        """
        dt = self.dt
        self.t += dt
        pos_t = self.pos

        # get previous dir in case there are no spikes in recent cycles and vector is None
        prev_dir = np.asarray(self.dir, dtype=float).reshape(2,) if self.dir is not None else None
        spike_ids, spike_counts = self.extract_late_phase_spikes(condition = condition, 
                                                   phase_threshold=phase_threshold, 
                                                   t_window=t_window, 
                                                   return_counts=return_counts,
                                                   min_time=getattr(self, 'decisionStartTime', None)) 
        decoded_dir = self.directionFromCurrentLateSpikes(pos_t, spike_ids, spike_counts)
        # store decoded direction for this time step
        self.decoded_dir_history.append(decoded_dir if decoded_dir is not None else np.array([np.nan, np.nan]))

        if decoded_dir is None: # if no spikes in the last few theta cycles, keep moving in the same direction as in the previous step
            if prev_dir is not None:
                self.dir = prev_dir 
            else: # if no previous direction (e.g. at the very start), default to moving to the right
                self.dir = np.array([1, 0])
        else: # if there are spikes, move in the direction decoded from those spikes
            self.dir = decoded_dir

        # DEBUG safety check if self.dir is nan or nonfinite (in case that agent runs in wrong direction after reset):
        if not np.isfinite(self.dir).all():
            print(f"Warning: used direction is not finite! Decoded dir: {decoded_dir}, Previous dir: {prev_dir}, self.dir before check: {self.dir}. Using previous direction if finite or default drection [1,0] for this step.")
            # handle this case by keeping previous direction or defaulting to [1,0] if no previous direction
            if prev_dir is not None and np.isfinite(prev_dir).all():
                self.dir = prev_dir
            else:
                self.dir = np.array([1, 0])

        if maxTurnAngle == True:
            if maxAngle is None:
                maxAngle = 90.0 # default to 90 degrees if maxTurnAngle is True but no maxAngle provided
            else: 
                maxAngle = float(maxAngle) # ensure it's a float
            
            # compute angle between prev_dir and decoded_dir via dot product + norm + compare to precomputed cos(maxAngle) 
            dot = float(np.clip(np.dot(prev_dir, self.dir), -1.0, 1.0)) # clip for numerical stability. direction vector is already normed
            dotAngleDeg = float(np.degrees(np.arccos(dot))) # get angle between previous direction and new direction in degrees
            if dotAngleDeg > maxAngle:
                # if angle btwn prev and new direction is larger than threshold angle, set new direction do be at max angle in direction of decoded direction
                angleSign = np.sign(np.cross(prev_dir, self.dir)) # get sign of angle via cross product (positive if decoded dir is to the left of prev dir, negative if to the right)
                angleSign = 1 if angleSign == 0 else angleSign # if exactly zero
                # compute new direction as prev_dir but turned by maxAngle in direction of decoded dir
                self.dir = turn(prev_dir, turnAngle=angleSign*maxAngle)

        # make sure self.dir is in the correct shape for the movement update
        if self.dir.shape != (2,): # (2,) looks like this: [0.5, 0.5] representing the x and y coordinates of the direction vector
            self.dir = self.dir.reshape(2,) # if not in that shape, reshape it to be so
        
        if self.mazeType == 'TMaze':
            # if in T maze, also need to check for junction and make left/right decision as in movementPolicyUpdate
            corridor_half_width = 0.05 * self.roomSize
            junction_x = self.roomSize + corridor_half_width
            stem_y = self.roomSize 
            lane_offset = 0.03 * self.roomSize
            
            # self.dir at junction decision [0,1] or [0,-1]   
            if (self.pos[0] > junction_x) and (self.LRDecisionPending == True):
                if self.dir[1] > 0: # if population vector is pointing upwards at junction, go into upper arm
                    self.dir = np.array([0, 1])
                    self.lastArmChoice = 1
                elif self.dir[1] < 0: # if population vector is pointing downwards at junction, go into lower arm
                    self.dir = np.array([0, -1])
                    self.lastArmChoice = -1
                else: # if population vector is pointing straight (unlikely but possible), choose same arm as last time or choose randomly if no last choice
                    print("straight at junction -> choosing arm based on last choice or randomly")
                    if hasattr(self, "lastArmChoice") and self.lastArmChoice is not None:
                        if self.lastArmChoice == 1:
                            self.dir = np.array([0, 1])
                        else:
                            self.dir = np.array([0, -1])
                    else:
                        if np.random.choice([0, 1]) == 0:
                            self.dir = np.array([0, 1])
                            self.lastArmChoice = 1
                        else:
                            self.dir = np.array([0, -1])
                            self.lastArmChoice = -1
                self.LRDecisionPending = False
                 
        # define proposed new position based on current position, direction and speed
        proposedNewPos = self.pos + self.speed * self.dir * dt
        proposedStep = np.array([self.pos,proposedNewPos])

        # check for collisions and update position and direction accordingly (using the same policies as in movementPolicyUpdate)
        checkResult = self.checkWallIntercepts(proposedStep)
        if self.movementPolicy == 'windowsScreensaver':
            if checkResult[0] == 'noImmediateCollision': # if no immediate collision, update position to proposed new position based on population vector direction
                self.pos = proposedNewPos
            elif checkResult[0] == 'collisionNow': # if immediate collision, bounce off wall so that agent can continue moving in same general direction but avoid getting stuck
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
            elif ((checkResult[0] == 'collisionAhead') and (self.biasWallFollow==True)): # if collision ahead and bias to follow wall is true, then follow wall so that agent can continue moving in same general direction but avoid getting stuck. difference 
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'follow')
        
        ### Teleport back to starting position when goal reached or midpoint of non-goal arm
        #if self.mazeType == 'TMaze':
            # agent has entered goal radius
            #if self.pos[1] > ((self.goalPos[1] - self.goalRadius)+0.5) and self.pos[0] > (self.goalPos[0] - self.goalRadius):
             #   print("Goal reached! Teleporting back to start.")
              #  self.pos = np.array([0.2, self.roomSize]) # reset position to start of T maze
               # self.dir = np.array([self.dir[0], self.dir[1]]) # keep direction as set by population vector, just pop back into maze
                #self.LRDecisionPending = True # allow new left/right decision at next junction
            # agent has entered midpoint of non-goal arm radius (i.e. made wrong choice at junction)
            #elif self.pos[1] < ((-self.goalPos[1] + self.goalRadius) - 0.5) and self.pos[0] > (self.goalPos[0] - self.goalRadius):
             #   print("Not reached goal, resetting position")
              #  self.pos = np.array([0.2, self.roomSize]) # reset position to start of T maze
               # self.dir = np.array([self.dir[0], self.dir[1]]) # keep direction as set by population vector, just pop back into maze
                #self.LRDecisionPending = True # allow new left/right decision at next junction

        # for wrap keep in mind that re-entry from arms to stem might cause issues with direction -> consider fixing if causes issue in behaviour
        #if self.mazeType == 'TMaze':
            #if self.enteredGoal(prevPos, self.pos):
            # Add closed-loop wrap from arms to stem 
            #if self.pos[1] > self.extent[3]: # if exited top arm
               # overshoot = self.pos[1] - self.extent[3] # calculate how much agent has overshot the end of the arm
                #self.pos = np.array([overshoot, stem_y + lane_offset]) # wrap around
                ## direction continues to be the direction of the population vector
                #self.dir = np.array([self.dir[0], self.dir[1]]) # coordinates remain as set by popiulation vector -> gotta see if that works in practice or if it causes weird behaviour
                #self.LRDecisionPending = True # allow new left/right decision at next junction
            #elif self.pos[1] < self.extent[2]: # if exited bottom arm
             #   overshoot = self.extent[2] - self.pos[1] # calculate how much agent has overshot the end of the arm
              #  self.pos = np.array([overshoot, stem_y - lane_offset]) # what is the lane offset: 
               # self.dir = np.array([self.dir[0], self.dir[1]]) # coordinates remain as set by popiulation vector -> gotta see if that works in practice or if it causes weird behaviour
                #self.LRDecisionPending = True # allow new left/right decision at next junction
        
        # catchcall instances a rat escapes the maze by accident, resets it back to the start position
        if ((self.pos[0] < self.extent[0]) or
            (self.pos[0] > self.extent[1]) or
            (self.pos[1] < self.extent[2]) or
            (self.pos[1] > self.extent[3])):
            print(self.pos)
            self.pos = np.array([0.2, self.roomSize]) # reset position to start of T maze
            #self.pos[0] = max(self.pos[0],self.extent[0]+0.02) # pop back in 2cm if escaped on left
            #self.pos[0] = min(self.pos[0],self.extent[1]-0.02) # pop back in 2cm if escaped on right
            #self.pos[1] = max(self.pos[1],self.extent[2]+0.02) # pop back in 2cm if escaped on bottom
            #self.pos[1] = min(self.pos[1],self.extent[3]-0.02) # pop back in 2cm if escaped on top
            print("Rat escaped! Reset to start.")
            if self.mazeType == 'TMaze':
                self.dir = np.array([self.dir[0], self.dir[1]]) # keep direction as set by population vector, just pop back into maze
                self.LRDecisionPending = True   


    def vectorsToCellCentres(self,pos,distance=False):
        """Takes a posisiton vector shape (2,) and returns an array of shape (nCells,2) of the 
        shortest vector path to all cells, taking into account loop geometry etc. 

        Args:
            pos (array): position vector shape (2,)

        Returns:
            vectorToCells (array): shape (30,2)
        """        
        if self.mazeType == 'loop' and self.doorsClosed == False:
            pos_plus = pos + np.array([self.roomSize,0])
            pos_minus = pos - np.array([self.roomSize,0])
            positions = np.array([pos,pos_plus,pos_minus])
            vectors = self.centres[:,np.newaxis,:] - positions[np.newaxis,:,:]
            shortest = np.argmin(np.linalg.norm(vectors,axis=-1),axis=1)
            shortest_vectors = np.diagonal(vectors[:,shortest,:],axis1=0,axis2=1).T

        else:
            shortest_vectors = self.centres - self.pos
            
        return shortest_vectors

    def distanceToCellCentres(self, pos):
        """Calculates distance to cell centres. 
           In the case of the two room maze, this distance is the shortest feasible walk carefully accounting for doorways etc. 
        Args:
            pos (no.array): The position to calculate the distances from 

        Returns:
            np.array: (nCells,) array of distances
        """         

        if self.mazeType == 'twoRooms': 
            distances = np.zeros(self.nCells)
            wall_x = self.walls['doors'][0][0][0]
            wall_y1, wall_y2 = self.walls['doors'][0][0][1], self.walls['doors'][0][1][1]
            for i in range(self.nCells):
                vec = np.array(pos - self.centres[i])
                if ((self.centres[i][0] < wall_x) and (pos[0] < wall_x)) or ((self.centres[i][0] > wall_x) and (pos[0] > wall_x)):
                    distances[i] = np.linalg.norm(vec)
                else: #cell and position in different rooms 
                    if self.doorsClosed == True:
                        distances[i] = 100*self.roomSize 
                        print("doorsClosed")
                    else:
                        step = np.array([pos,self.centres[i]])
                        if self.checkWallIntercepts(step)[0] == 'collisionNow':
                            pastBottomWall = np.linalg.norm(np.array([wall_x,wall_y1]) - pos) + np.linalg.norm(np.array([wall_x,wall_y1]) - self.centres[i])
                            pastTopWall = np.linalg.norm(np.array([wall_x,wall_y2]) - pos) + np.linalg.norm(np.array([wall_x,wall_y2]) - self.centres[i])
                            distances[i] = min(pastBottomWall,pastTopWall)
                        else: 
                            distances[i] = np.linalg.norm(vec)

        else: 
            shortest_vector = self.vectorsToCellCentres(pos)
            distances = np.linalg.norm(shortest_vector,axis=1)

        return distances


    def toggleDoors(self, doorsClosed = None): #this function could be made more advanced to toggle more maze options
        """Opens or closes door and updates mazeState
            mazeState stores the most recent version of the maze walls dictionary which will include 'door' wall only if doorsClosed is True
        Args:
            doorsClosed ([bool], optional): True is doors to be closed, False if doors to be opened. Defaults to None, in which case current door state is flipped.
        Returns:
            [dict]: the walls dictionary
        """        
        if doorsClosed is not None: 
            self.doorsClosed = doorsClosed
        else: self.doorsClosed = not self.doorsClosed

        walls = self.walls.copy()
        if self.doorsClosed == False: 
            del walls['doors']
            self.mazeState['walls'] = walls
        elif self.doorsClosed == True: 
            self.mazeState['walls'] = walls

        self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 

        return self.mazeState['walls']

    def checkWallIntercepts(self,proposedStep,collisionDistance=0.1): #proposedStep = [pos,proposedNextPos]
        """Given the cuurent proposed step [currentPos, nextPos] it calculates whether a collision with any of the walls exists along this step.
        There are three possibilities from most worrying to least:
            • there is a collision ON the current step. Do something immediately.
            • there is a collision along the current trajectory in the next few cm's, but not on the current step. Consider doing something.
            • there is no collision coming up soon. Carry on as you are. 
        Args:
            proposedStep (array): The proposed step. np.array( [ [x_current, y_current] , [x_next, y_next] ] )

        Returns:
            tuple: (str, array), (<whether there is no collision, collision now or collision ahead> , <the wall in question>)
        """        
        s1, s2 = np.array(proposedStep[0]), np.array(proposedStep[1])
        pos = s1
        ds = s2 - s1
        stepLength = np.linalg.norm(ds)
        ds_perp = perp(ds)

        collisionList = [[],[]]
        futureCollisionList = [[],[]]

        #check if the current step results in a collision 
        walls = self.mazeState['walls'] #current wall state

        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                w1, w2 = np.array(wall[0]), np.array(wall[1])
                dw = w2 - w1
                dw_perp = perp(dw)

                # calculates point of intercept between the line passing along the current step direction and the lines passing along the walls,
                # if this intercept lies on the current step and on the current wall (0 < lam_s < 1, 0 < lam_w < 1) this implies a "collision" 
                # if it lies ahead of the current step and on the current wall (lam_s > 1, 0 < lam_w < 1) then we should "veer" away from this wall
                # this occurs iff the solution to s1 + lam_s*(s2-s1) = w1 + lam_w*(w2 - w1) satisfies 0 <= lam_s & lam_w <= 1
                with np.errstate(divide='ignore'):
                    lam_s = (np.dot(w1, dw_perp) - np.dot(s1, dw_perp)) / (np.dot(ds, dw_perp))
                    lam_w = (np.dot(s1, ds_perp) - np.dot(w1, ds_perp)) / (np.dot(dw, ds_perp))

                #there are two situations we need to worry about: 
                # • 0 < lam_s < 1 and 0 < lam_w < 1: the collision is ON the current proposed step . Do something immediately.
                # • lam_s > 1     and 0 < lam_w < 1: the collision is on the current trajectory, some time in the future. Maybe do something. 
                if (0 <= lam_s <= 1) and (0 <= lam_w <= 1):
                    collisionList[0].append(wall)
                    collisionList[1].append([lam_s,lam_w])
                    continue

                if (lam_s > 1) and (0 <= lam_w <= 1):
                    if lam_s * stepLength <= collisionDistance: #if the future collision is under collisionDistance away
                        futureCollisionList[0].append(wall)
                        futureCollisionList[1].append([lam_s,lam_w])
                        continue
        
        if len(collisionList[0]) != 0:
            wall_id = np.argmin(np.array(collisionList[1])[:,0]) #first wall you collide with on step 
            wall = collisionList[0][wall_id]
            return ('collisionNow', wall)
        
        elif len(futureCollisionList[0]) != 0:
            wall_id = np.argmin(np.array(futureCollisionList[1])[:,0]) #first wall you would collide with along current step 
            wall = futureCollisionList[0][wall_id]
            return ('collisionAhead', wall)
        
        else:
            return ('noImmediateCollision',None)

    def getPlaceFields(self, M=None, threshold=None):
        """Calculates receptive fiels of all place cells 
            There is one place cell for each feature cell. 
            A place cell (as  in de Cothi 2020) is defined as a thresholded linear combination of feature cells
            where the linear combination is a row of the SR matrix. 
        Args:
            M (array): SR matrix
        Returns:
            array: Receptive fields of shape [nCells, nX, nY]
        """        
        if M is None: 
            M = self.M
        M = M.copy()
        #normalise: 
        # M = M / np.diag(M)[:,np.newaxis]
        if threshold is None: 
            placeCellThreshold =  0.9  #place cell threshold value (fraction of its maximum)
        else: 
            placeCellThreshold =  threshold
        placeFields = np.einsum("ij,klj->ikl",M,self.discreteStates)
        threshold = placeCellThreshold*np.amax(placeFields,axis=(1,2))[:,None,None]
        # threshold = placeCellThreshold
        placeFields = np.maximum(0,placeFields - threshold)
        return placeFields

    def getGridFields(self, M, alignToFinal=False):
        """Calculates receptive fiels of all grid cells 
            There is an equal number of grid cells as place cells and feature cells. 
            A grid cell (as in de Cothi 2020) is defined as a thresholded linear combination of feature cells
            where the linear combination weights are the eigenvectors of the SR matrix. 
        Args:
            M (array): SR matrix
            alignToFinal (bool): Since negative of eigenvec is also eigenvec try maximise overlap with final one (for making animations)
        Returns:
            array: Receptive fields of shape [nCells, nX, nY]
        """
        M = M.copy()
        _, eigvecs = np.linalg.eig(M) #"v[:,i] is the eigenvector corresponding to the eigenvalue w[i]"
        eigvecs = np.real(eigvecs)
        gridCellThreshold = 0 
        gridFields = np.einsum("ij,kli->jkl",eigvecs,self.discreteStates)
        threshold = gridCellThreshold*np.amax(gridFields,axis=(1,2))[:,None,None]
        if alignToFinal == True:
            grids_final_flat = np.reshape(self.gridFields,(self.stateSize,-1))
            grids_flat = np.reshape(gridFields,(self.stateSize,-1))
            dotprods = np.empty(grids_flat.shape[0])
            for i in range(len(dotprods)):
                dotprodsigns = np.sign(np.diag(np.matmul(grids_final_flat,grids_flat.T)))
                gridFields *= dotprodsigns[:,None,None]
        gridFields = np.maximum(0,gridFields)
        return gridFields
    
    def posToState(self, pos, stateType=None, normalise=True, cheapNormalise=False,initialisingCells=False): #pos is an [n1, n2, n3, ...., 2] array of 2D positions
        
        if (self.statesAlreadyInitialised == False) or (self.firingRateLookUp == False):
            #calculates the firing rate of all cells 
            pos = np.array(pos)
            if stateType == None: stateType = self.stateType
        
            vector_to_cells = self.centres - pos
            distance_to_cells = [np.linalg.norm(vector_to_cells,axis=1)]
            closest_cell_ID = np.argmin(distance_to_cells)

            if (self.mazeType == 'loop') and (self.doorsClosed == False):
                distance_to_cells.append(np.linalg.norm(self.centres - pos + [self.extent[1],0],axis=1))
                distance_to_cells.append(np.linalg.norm(self.centres - pos - [self.extent[1],0],axis=1))
            
            if (self.mazeType == 'twoRooms'): 
                distance_to_cells = [self.distanceToCellCentres(pos)]

            if stateType == 'onehot':
                state = np.zeros(self.nCells)
                state[closest_cell_ID] = 1
            
            if stateType == 'gaussianThreshold':
                state = np.zeros(self.nCells)
                for distance in distance_to_cells: 
                    state += np.maximum(np.exp(-distance**2 / (2*(self.sigmas**2))) - np.exp(-1/2) , 0) / (1-np.exp(-1/2))
                    # state = state / (self.sigmas) #normalises so same no. spikes emitted for all cell sizes

            if stateType == 'gaussian':
                state = np.zeros(self.nCells)
                for distance in distance_to_cells: 
                    state += np.exp(-distance**2 / (2*(self.sigmas**2)))
                    state = state/self.sigmas

            if stateType == 'bump':
                state = np.zeros(self.nCells)
                for distance in distance_to_cells:
                    state[distance<self.sigmas] += np.e * np.exp(-1/(1-(distance/self.sigmas)**2))[distance<self.sigmas]
                    state[distance>=self.sigmas] += 0
                    state = state/self.sigmas
        
        else:
            #uses look up table to rapidly pull out the firing rates without having to recalculate them 

            closestQuantisedArea = np.unravel_index(np.argmin(np.linalg.norm(self.discreteCoords - pos,axis=-1)),self.discreteCoords.shape[:-1])
            state = self.discreteStates[closestQuantisedArea]


        return state


    def positionArray_to_stateArray(self, positionArray, stateType=None,verbose=False): 
        """Takes an array of 2D positions of size (n1, n2, n3, ..., 2)
        returns the state vector for each of these positions of size (n1, n2, n3, ..., N) where N is the size of the state vector
        Args:
            positionArray ([type]): [description]
            stateType ([type], optional): [description]. Defaults to None.
        """        
        if stateType == None: stateType = self.stateType
        states = np.zeros(positionArray.shape[:-1] + (self.nCells,))
        if verbose == False:
            for idx in np.ndindex(positionArray.shape[:-1]):
                states[idx] = self.posToState(pos = positionArray[idx],stateType = stateType)
        else: 
            for idx in tqdm(np.ndindex(positionArray.shape[:-1]),total=int(positionArray.size/2)):
                states[idx] = self.posToState(pos = positionArray[idx],stateType = stateType)

        return states
    
    def averageM(self, M=None):
        if M == None:
            M = self.M
        M_copy = M.copy()
        roll = int(self.nCells/2)
        for i in range(self.nCells): # Livi: changed from agent.nCells to self.nCells bcs latter doesn't give erorr
            M_copy[i,:] = np.roll(M[i,:],-i+roll)
        M_av,M_std = np.mean(M_copy,axis=0),np.std(M_copy,axis=0)
        M_av, M_std = M_av/np.max(M_av), M_std/np.max(M_std)
        return M_av, M_std
    
    def getMetrics(self,time=None):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time).abs().to_numpy().argmin()
            snapshot = self.snapshots.iloc[hist_id]
        else:
            snapshot = self.snapshots.iloc[-1]

        x = self.centres[:,0].copy()

        W = rowAlignMatrix(snapshot['W'].copy())
        W_notheta = rowAlignMatrix(snapshot['W_notheta'].copy())
        M = rowAlignMatrix(self.snapshots.iloc[-1]['M'].copy())
        W_scrambled = rowAlignMatrix(snapshot['W_scrambled'].copy()) # Livi addition

        mid = int(self.nCells / 2)

        #R2s
        R_W = Rsquared(W,M)              
        R_Wnotheta = Rsquared(W_notheta,M)  
        R_Wscrambled = Rsquared(W_scrambled,M) # Livi addition

        #SNRs
        SNR_W = (np.max(np.mean(W,axis=0)) - np.min(np.mean(W,axis=0))) / np.mean(np.std(W,axis=0)[mid-5:mid+5])
        SNR_Wnotheta = (np.max(np.mean(W_notheta,axis=0)) - np.min(np.mean(W_notheta,axis=0))) / np.mean(np.std(W_notheta,axis=0)[mid-5:mid+5])
        SNR_Wscrambled = (np.max(np.mean(W_scrambled,axis=0)) - np.min(np.mean(W_scrambled,axis=0))) / np.mean(np.std(W_scrambled,axis=0)[mid-5:mid+5])

        #skews
        W_flat = np.mean(W,axis=0)/np.trapezoid(np.mean(W,axis=0),x)
        W_flat /= np.max(W_flat)
        Wnotheta_flat = np.mean(W_notheta,axis=0)/np.trapezoid(np.mean(W_notheta,axis=0),x)
        Wnotheta_flat /= np.max(Wnotheta_flat)
        M_flat = np.mean(M,axis=0)/np.trapezoid(np.mean(M,axis=0),x)
        M_flat /= np.max(M_flat)
        # add scramble(Livi addition)
        W_scrambled_flat = np.mean(W_scrambled,axis=0)/np.trapezoid(np.mean(W_scrambled,axis=0),x)
        W_scrambled_flat /= np.max(W_scrambled_flat)    

        try: skew_W = getSkewness(W_flat)
        except RuntimeError: skew_W = np.NaN
        try: skew_Wnotheta = getSkewness(Wnotheta_flat)
        except RuntimeError: skew_Wnotheta = np.NaN
        try: skew_M = getSkewness(M_flat)
        except RuntimeError: skew_M = np.NaN
        # add scramble(Livi addition)
        try: skew_Wscrambled = getSkewness(W_scrambled_flat)
        except RuntimeError: skew_Wscrambled = np.NaN

        #peaks 
        peak_W = x[np.argmax(W_flat)] - 2.5
        peak_Wnotheta = x[np.argmax(Wnotheta_flat)] - 2.5
        peak_M = x[np.argmax(M_flat)] - 2.5
        peak_Wscrambled = x[np.argmax(W_scrambled_flat)] - 2.5 # Livi addition
        return R_W, R_Wnotheta, R_Wscrambled, SNR_W, SNR_Wnotheta, SNR_Wscrambled, float(skew_W), float(skew_Wnotheta), float(skew_M), float(skew_Wscrambled), peak_W, peak_Wnotheta, peak_M, peak_Wscrambled

    def saveToFile(self,name,directory="../savedObjects/"):
        np.savez(directory+name+".npz",self.__dict__)
        return

    def loadFromFile(self,name,directory="../savedObjects/"):
        attributes_dictionary = np.load(directory+name+".npz",allow_pickle=True)['arr_0'].item()
        print("Loading attributes...",end="")
        for key, value in attributes_dictionary.items():
            setattr(self, key, value)
        print("done. use 'agent.__dict__.keys()'  to see available attributes")
    
    def setGoal(self, goalPos=None, goalRadius=None, goalReward=None):
        """Sets the goal position, radius and reward for the agent.
        Args:
            goalPos (array, optional): 2D position of the goal. Defaults to None.
            goalRadius (float, optional): Radius around the goal position within which the agent receives the reward. Defaults to None.
            goalReward (float, optional): Reward received when the agent is within the goal radius.
        Returns:            None
        """
        if goalPos is None:
            self.goalPos = np.array([self.roomSize*0.5,self.roomSize*0.5]) #default at centre of room
        else:
            self.goalPos = goalPos
        if goalRadius is None:
            self.goalRadius = 0.2 #default radius of 20cm
        else:
            self.goalRadius = goalRadius
        if goalReward is None:
            self.goalReward = 1.0 #default reward of 1
        else:
            self.goalReward = goalReward
        return

    def enteredGoal(self, prevPos, newPos):
        """Checks if the agent has entered the goal area between the previous and new positions.
        Args:
            prevPos (array): Previous 2D position of the agent.
            newPos (array): New 2D position of the agent.
        Returns:
            bool: True if the agent has entered the goal area, False otherwise.
        """
        if self.goalPos is None or self.goalRadius is None:
            print("Goal position and radius not set. Set both of them first!")
            return False
        
        prevPos = np.asarray(prevPos, dtype=float).reshape(2,)
        newPos = np.asarray(newPos, dtype=float).reshape(2,)
        goalC = np.asarray(self.goalPos, dtype=float).reshape(2,)
        goalR = float(self.goalRadius)

        # distance from previous position and new position to goal centre and whether distance >< goal radius
        prevIn = np.linalg.norm(prevPos - goalC) <= goalR
        newIn = np.linalg.norm(newPos - goalC) <= goalR

        # standard case: agent entred goal radius from outside
        if (not prevIn) and newIn:
            return True
        # Crossing case: not defiend for now bcs radius will never be that small

        # count only as an entry event if agent was outside goal radius in previous step and is now inside goal radius. If agent was already inside goal radius in previous step, we don't want to count that as a new entry.
        else:
            return False

    ## Spike Decoder Functions
    def get_phase_at_time(self, t, condition="theta"):
        """
        Get the global theta phase at time t for a given condition.
        Later used to filter spikes based on their phase at the time they occurred.
        
        Args:
            t: time point
            condition: 'theta', 'notheta', or 'scrambled'
        
        Returns:
            phase in [0, 2π]
        """
        if condition == "theta":
            return np.mod(2 * np.pi * self.thetaFreq * t, 2 * np.pi)
        elif condition == "notheta":
            return np.pi / 2  # Constant phase (peak of theta)
        elif condition == "scrambled":
            # Use time-stamped scrambled phase history; this is robust when history is not sampled at every dt.
            if hasattr(self, 'thetaPhase_scrambled_history') and len(self.thetaPhase_scrambled_history) > 0:
                if hasattr(self, 'thetaPhase_scrambled_time_history') and len(self.thetaPhase_scrambled_time_history) == len(self.thetaPhase_scrambled_history):
                    times = np.asarray(self.thetaPhase_scrambled_time_history, dtype=float).reshape(-1)
                    idx = int(np.argmin(np.abs(times - float(t))))
                else:
                    # Backward-compatible fallback for objects without timestamped phase history.
                    idx = int(np.clip(np.round(float(t) / self.dt), 0, len(self.thetaPhase_scrambled_history) - 1))
                phase_val = float(np.asarray(self.thetaPhase_scrambled_history[idx]).reshape(-1)[0])
                return np.mod(phase_val, 2 * np.pi)

            if not hasattr(self, '_scrambled_phase_fallback_warned'):
                self._scrambled_phase_fallback_warned = False
            if self._scrambled_phase_fallback_warned is False:
                print("Using clean theta as fallback for scrambled phase.")
                self._scrambled_phase_fallback_warned = True
            return np.mod(2 * np.pi * self.thetaFreq * t, 2 * np.pi)
        else:
            print("Unknown condition. Using constant phase.")
            return np.pi / 2 # default to constant phase if unknown condition
        
    def extract_late_phase_spikes(self, condition="theta", phase_threshold=1.0 * np.pi, t_window=None, return_counts=True, min_time=None):
        """
        Extract cells that fired at late theta phase within a recent time window.
        
        Args:
            condition: 'theta', 'notheta', or 'scrambled'
            phase_threshold: phase value (radians) above which spikes count as "late"
            t_window: time window size (seconds). If None, uses last 2 theta cycles.
        
        Returns:
            late_phase_cell_ids: list of cell IDs that fired at late phase
        """
        # default time window: 2 theta cycles -> 2:10 = 0.2s for 10Hz theta => one cycle every 0.1s, so 2 cycles = 0.2s
        if t_window is None:
            # Default: 2 theta cycles
            t_window = 2 / self.thetaFreq
        
        t_start = max(0, self.t - t_window) # max with 0 is to ensure we don't have negative start time at beginning of simulation
        if min_time is not None:
            t_start = max(t_start, float(min_time))
        t_end = self.t
        
        # Get spike data for this condition (CA3 or CA1; using CA3 here)
        try:
            if condition not in self.spikedata_by_condition:
                return ([], []) if return_counts else []
            spike_data = self.spikedata_by_condition[condition]["CA1"]
        except (KeyError, AttributeError):
            return ([], []) if return_counts else []
        
        # Extract spike times and IDs from spike data dictionary for CA3 or CA1
        if isinstance(spike_data, dict) and "times" in spike_data and "ids" in spike_data:
            spike_times = np.asarray(spike_data["times"], dtype=float).reshape(-1) # reshape to ensure it's a 1D array bcs otherwise it can cause issues with the boolean masking later on if it's a 2D array with one column
            spike_ids = np.asarray(spike_data["ids"], dtype=int).reshape(-1)
        else:
            return ([], []) if return_counts else []
        
        # Filter spikes in time window
        mask_time = (spike_times >= t_start) & (spike_times <= t_end) # create a mask to filter spikes that occurred within the time window
        spikes_in_window = spike_ids[mask_time] # get IDs of neurons that fired within that time window
        spike_times_in_window = spike_times[mask_time] # get times of the spikes that occurred within that window 
        
        if len(spikes_in_window) == 0: # if no spikes in window, return empty list
            return ([], []) if return_counts else []
        
        # Get phase at each spike time -> determine which spikes are late phase
        phases_at_spikes = np.array([self.get_phase_at_time(st, condition=condition) # get the phase at the time of each spike in the window
                                      for st in spike_times_in_window]) # create an array of the same length as spikes_in_window where each entry is the phase at the time of that spike    
        
        # now we got IDs of neurons spiking in time window and times of spikes as well as the phase at which spikes occured. Next we filter to only keep the spikes that were at late phases 
        # Keep only late-phase spikes
        late_phase_mask = phases_at_spikes >= phase_threshold #create mask with set threshold
        late_phase_cell_ids = spikes_in_window[late_phase_mask]

        if return_counts == True:
            unique_ids, counts = np.unique(late_phase_cell_ids, return_counts=True)
            return unique_ids.tolist(), counts.astype(int).tolist() # return both unique cell IDs and their corresponding counts as lists
        else:
            return list(np.unique(late_phase_cell_ids)) # unique means if a cell fired multiple late-phase spikes, it only counts once for decoding -> this way we get a set of active cells at late phase rather than a count of spikes which could be dominated by a few very active cells.
        

    def _get_position_at_time(self, t):
        """Return agent position at time t using nearest-neighbour lookup in history."""
        if self.history is None or len(self.history) == 0:
            return None
        hist_times = self.history['t'].to_numpy(dtype=float) # convert history times to numpy array for efficient processing
        idx = int(np.argmin(np.abs(hist_times - t))) # find index of closest time point in history to the requested time t -> this is bcs we discretise time in the history and may not have an entry for exactly t, so we find the closest one. This is a common approach for time-series data.
        return np.array(self.history.iloc[idx]['pos'], dtype=float) # iloc is used to access the row at the index and we extract the 'pos' column from that row -> gives us position at that time point

    def _decode_direction_from_spikes(self, pos_t, spike_ids):
        """Decode heading vector from active spike IDs using vectors to place-cell centres.
        Args:
            pos_t: position at time t (array of shape (2,))
            spike_ids: list or array of active cell IDs at time t, taken from late-phase spikes
        Returns:
            decoded_direction: unit vector representing decoded heading direction, or None if decoding fails (e.g. no valid spikes or zero vector)
        """
        if pos_t is None or spike_ids is None or len(spike_ids) == 0:
            return None

        spike_ids = np.asarray(spike_ids, dtype=int) 
        valid = (spike_ids >= 0) & (spike_ids < self.nCells) # create a boolean mask to filter out any spike IDs that are out of bounds (negative or greater than number of cells)
        spike_ids = spike_ids[valid] # apply the mask to keep only valid spike IDs
        if spike_ids.size == 0: # if after filtering there are no valid spike IDs left, return None
            return None

        # Weight by spike count so repeatedly active cells contribute more.
        unique_ids, counts = np.unique(spike_ids, return_counts=True) # get unique cell IDs and how many times each fired in the late phase window -> this allows us to weight the contribution of each cell by how active it was in that window, which can improve decoding accuracy by giving more influence to cells that were more active.
        vectors = self.centres[unique_ids] - pos_t[np.newaxis, :] # get vectors from current position to centres of active cells -> gives sense of where each active cell's place field is relative to agent's current position 
        weighted = vectors * counts[:, np.newaxis] # weight each vector by the count of spikes for that cell -> this way cells that fired more contribute more to the decoded direction
        decoded = np.sum(weighted, axis=0) # sum the weighted vectors to get a single vector representing the overall decoded direction based on the active cells -> this is essentially a population vector decoding approach where we combine the contributions of all active cells to get a single estimate of direction
        norm = np.linalg.norm(decoded) # this is what norm s doing: it calculates the length of the decoded vector. If this length is very small (close to zero), it means that the active cells' contributions are cancelling each other out and we don't have a clear direction signal, so we return None. If the norm is sufficiently large, we normalize the decoded vector to get a unit vector representing the decoded heading direction.
        if norm < 1e-12: 
            return None
        return decoded / norm # return the decoded direction as a unit vector by dividing by its norm

    def directionFromCurrentLateSpikes(self, pos_t, spike_ids, spike_counts=None, normalise = True):
        """
        Compute a direction/planned trajectory from the late phase spikes within a recent time window from the current position of the agent.
        Args:
            pos_t: position at current time t (array of shape (2,), i.e the x and y coordinates of the agent at time t)
            spike_ids: list or array of active cell IDs at time t, taken from late-phase spikes
        Returns:
            direction vector (array of shape (2,)) representing the decoded heading direction based on the late-phase spikes, or None if decoding fails
        """
        # get a vector from current position of agent along the trajectory that is planned in the late phase spikes
        # get the IDs of late phase cells that fired within recent time window and the agent's position at current time t
        if pos_t is None or spike_ids is None or len(spike_ids) == 0:
            return None
        
        # pos_t is the current position of the agent at time t, which ca be defined in the policy loop later on I guess     
        pos_t = np.asarray(pos_t, dtype=float).reshape(-1) # reshape to ensure it's a 1D array of shape (2,) representing x and y coordinates of the agent's position at time t

        spike_ids = np.asarray(spike_ids, dtype=int) # ensure spike IDs are in a numpy array for efficient processing
        valid = (spike_ids >= 0) & (spike_ids < self.nCells) # create a boolean mask to filter out any spike IDs that are out of bounds (negative or greater than number of cells)
        spike_ids = spike_ids[valid] # filter for spikes that are out of bounds
        if spike_ids.size == 0: # if after filtering there are no valid spike IDs
            return None

        if spike_counts is None:
            unique_ids, counts = np.unique(spike_ids, return_counts=True) # get unique cell IDs and how many times each fired in the late phase window -> this allows us to weight the contribution of each cell by how active it was in that window, which can improve decoding accuracy by giving more influence to cells that were more active.
            counts = counts.astype(float)
        else:
            counts = np.asarray(spike_counts, dtype=float).reshape(-1)
            if counts.size != np.asarray(spike_ids).size:
                 return None # if spike counts are provided, they must match the size of spike IDs
            unique_ids = spike_ids
            positive = counts > 0
            unique_ids = unique_ids[positive]
            counts = counts[positive]
            if unique_ids.size == 0: # if after filtering for positive counts there are no
                return None
        
        vectors = self.centres[unique_ids]- pos_t[np.newaxis, :] # get vectors from current position to centres of active cells -> gives sense of where each active cell's place field is relative to agent's current position.
        
        ### CONSIDER NORMALISING VECTORS SO ALL VECTORS CONTRIBUTE EQUALLY REGARDLESS OF DISTANCE TO PLACE FIELD CENTRE -> RN FURTHER CELLS CONTRIBUTE MORE
        if normalise == True:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) # calculate the length of each vector from current position to active cell centres. keepdims=True keeps the dimensions of the array so that we can divide by norms later on without issues with broadcasting. This gives us an array of shape (nActiveCells, 1) where each entry is the norm of the corresponding vector.
            norms[norms < 1e-12] = 1e-12 # to avoid division by zero, we set any very small norms to a small positive value
            vectors = vectors / norms # normalise each vector to have length 1 so that all cells contribute equally to the decoded direction regardless of how far their place field is from the agent's current position. This way we focus on the direction information rather than the magnitude of the vectors which could be dominated by cells with place fields close to the agent.
    
        weighted = vectors * counts[:, np.newaxis] # weight each vector by the count of spikes for that cell -> this way cells that fired more contribute more to the decoded direction
        ### INSERT VALUE COMPUTATION HERE
        valuedVectors = self.getValue(weightedVectors=weighted, unique_ids=unique_ids, alpha=1.0, beta=1.0) # get the value-weighted vectors based on the cell centre values -> this allows us to give more weight to vectors pointing towards more valuable locations (e.g. closer to reward) and less weight to vectors pointing towards less valuable locations, which can improve the relevance of the decoded direction for guiding behavior towards rewards
        decoded = np.sum(valuedVectors, axis=0) # sum the weighted vectors to get a single population vector that represents the overall direction based on active late phase cells. axis =0 means that 
        norm = np.linalg.norm(decoded) # calculate the length of the decoded vector. If this
        if norm < 1e-12: # i.e. if there is no clear direction signal
            return None
        
        return decoded/norm # return as unit vector of length 1 so speed is not affected by magnitude of vector

    def getValue(self,weightedVectors, unique_ids, alpha=1.0, beta=None):
        """
        The value function of the model. This function defines how valuable individual direction vectors are based on 
        the value of the cell centre they point at. Cells closer to the rewards have a higher value, meaning that vectors 
        pointing towards themj will have a higher value and contribute more to the direction of the population vector.
        This is implemented with a simple function where the vector is multiplied by a factor (a + V(ci)), where a is a baseline
        value preventing the magnitude of a vector from being zero when the cell is still far away from the reward but yet points towards it,
        and V(ci) is the value of the cell centre that the vector points towards. 
        Args:
            wVectors (array): Vectors from current position to active cells, weighted by spike counts 
            alpha (float): Baseline value for the value function
            beta (float): Coefficient for the cell centre value in the value function
        Returns:
            valuedVectors (array): The input vectors weighted by their value based on the cell centres they point towards
        """
        if alpha is None:
            alpha = 1.0
        else:
            alpha = float(alpha)
        if beta is None:
            beta = 1.0
        else:
            beta = float(beta)

        # get cell centre values 
        #cellCentres = self.centres
        cellValues = self.cellValues[unique_ids] # get precomputed cell values based on distance to goal -> array of shape (nCells, ) where each entry is value of corresponding cell centre. => so we don't compute values every time we calland reduce comp. costs
        weightedVectors = np.asarray(weightedVectors, dtype=float) # ensure weightedVectors is a numpy array for efficient processing
        if weightedVectors.ndim != 2 or weightedVectors.shape[1] != 2:
            raise ValueError("weightedVectors must be an array of shape (nVectors, 2) representing 2D vectors.")

        # compute valued vectors
        valuedVectors = weightedVectors * (alpha + beta * cellValues[:, np.newaxis]) # multiply each vector by its value based on the cell centre it points towards. cellValues[:, np.newaxis] reshapes cellValues to be a column vector so that it can be broadcasted correctly when multiplying with weightedVectors
        return valuedVectors
    
    def getCellValues(self, sigma = 3.0, manualCutoff = True):
        """
        Get the values for each cell centre. In this implementeation, the value of a cell is defined by the inverse of its distance to the goal, so cells closer to the goal have higher values.
        This is a simple way to implement a value function that encourages the agent to move towards the goal by giving more Value to vector pointing towards cells that are closer to the goal. 
        Returns:
            cellValues (array): An array of shape (nCells,) where each entry is the value of the corresponding cell centre based on its distance to the goal. 
        """
        ### NOTE: BEST TO PRECOMPUTE THIS IN THE INIT FUNCTION AND THEN JUST PULL OUT THE VALUES HERE RATHER THAN RECOMPUTING EVERY TIME WE CALL THIS FUNCTION, ESPECIALLY IF WE HAVE A LARGE NUMBER OF CELLS -> CAN JUST UPDATE THE VALUES IF THE GOAL CHANGES
        # get goal position
        goalPos = self.goalPos
        if goalPos is None:
            raise ValueError("Goal position not set. Please set the goal position in the agent parameters.")
        
        # compute distance from each cell centre to goal position
        cellCentres = self.centres
        distances = np.linalg.norm(cellCentres - goalPos, axis=1) # np.linalg.norm with axis=1 computes the Euclidean distance from each cell centre to the goal position. euclidean distance does not take into account any obstacles tho, such as walls
        sigma = float(sigma)
        # compute value of cell based on distance to goal -> but 0<=value<=1 so we can easily combine with baseline alpha in value function. We use a simple inverse relationship where value is higher for cells closer to the goal, but we also add a small constant to the denominator to prevent division by zero for cells that are very close to the goal.
        # values decrease not linearly with distance to goal, but exponentially, so that cells that are very close to the goal have much higher values than cells that are further away, which can create a stronger gradient for the agent to follow towards the goal.
        # exponential decay. NOTE: Could also try Gaussian decay fro smoother approach of 1 at goal and more gradual decrease with distance
        cellValues  = np.exp(-distances/sigma) 
        # Gaussian decay alternative:
        #cellValues = np.exp(-((distances**2)/2*sigma**2))# cells close to goal have values close to 1 and cells further away have values close to zero, but nver exactly zero
        cellValues = cellValues / np.max(cellValues) # normalise to max of 1 so that values are between 0 and 1
        # set values of cells in right arm to zero to indicate that that arm is wrong direction
        if manualCutoff == True:
            #select cells in right arm, i.e. cells 80- 119
            cellValues[80:120] = 0.0
            
        return cellValues


    def validate_late_phase_decoder(self,
                                    condition="theta",
                                    phase_threshold=1.5 * np.pi,
                                    t_window=None,
                                    future_window=0.1,
                                    sample_every=0.05,
                                    min_late_spikes=1,
                                    return_timeseries=False):
        """
        Decode-only validation: compares late-phase decoded heading vs actual future heading.

        This does NOT control movement. It only evaluates whether late-phase spikes contain
        predictive direction information.

        Args:
            condition: 'theta', 'notheta', or 'scrambled'.
            phase_threshold: Phase cutoff for defining late-phase spikes.
            t_window: Backward window for collecting spikes at each decode time.
            future_window: Time offset (s) used to compute the true future heading.
            sample_every: Temporal spacing (s) between decode evaluations.
            min_late_spikes: Minimum late-phase spikes required to produce a decode.
            return_timeseries: If True, include per-timepoint DataFrame.

        Returns:
            dict with coverage, angular error stats, and directional correlation.
        """
        if t_window is None:
            t_window = 2 / self.thetaFreq

        if self.history is None or len(self.history) < 3:
            return {
                'condition': condition,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': 'Not enough trajectory history to evaluate decoder.'
            }

        if not hasattr(self, 'spikedata_by_condition'):
            return {
                'condition': condition,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': 'No condition-specific spike data found. Use full STDP rule to record per-condition spikes.'
            }

        if condition not in self.spikedata_by_condition:
            return {
                'condition': condition,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': f'Condition {condition} not found in spike data.'
            }

        spike_dict = self.spikedata_by_condition[condition].get('CA1', {'times': [], 'ids': []})
        spike_times = np.asarray(spike_dict.get('times', []), dtype=float).reshape(-1)
        spike_ids = np.asarray(spike_dict.get('ids', []), dtype=int).reshape(-1)

        if spike_times.size == 0:
            return {
                'condition': condition,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': 'No spikes available for this condition.'
            }

        t_min = float(self.history['t'].min()) + t_window # need to look back this far to get spikes for decoding
        t_max = float(self.history['t'].max()) - future_window # to look ahead where movement actually went
        if t_max <= t_min:
            return {
                'condition': condition,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': 'Time span too short for requested windows.'
            }

        eval_times = np.arange(t_min, t_max, sample_every) # create array of time points at which to evaluate the decoder, spaced by sample_every seconds, starting from t_min to t_max -> so decodes every 0.05s by default but only within the time range where we have enough history to look back for spikes and enough future to compute true heading
        if eval_times.size == 0: # if the range is too short for the given sample_every, we can still do one evaluation at the midpoint of the range
            eval_times = np.array([(t_min + t_max) / 2.0])

        rows = []
        for t_eval in eval_times:
            pos_now = self._get_position_at_time(t_eval)
            if pos_now is None:
                rows.append((t_eval, False, np.nan, np.nan, 0, np.nan, np.nan))
                continue
            pos_x = float(pos_now[0])
            pos_y = float(pos_now[1])

            t0 = t_eval - t_window
            t1 = t_eval # the "current" time at which we want to decode the heading based on recent spikes
            in_window = (spike_times >= t0) & (spike_times <= t1) # create a boolean mask to filter spikes that occurred within the time window leading up to t_eval
            if not np.any(in_window): # if no spikes in this window, we can't decode, so we record this time point as not decoded and move on
                rows.append((t_eval, False, np.nan, np.nan, 0, pos_x, pos_y))
                continue

            sw_times = spike_times[in_window] # get only spike times that are within the window
            sw_ids = spike_ids[in_window] # get corresponding spike IDs for those times -> these are the spikes we will consider for decoding at this time point
            sw_phases = np.array([self.get_phase_at_time(ts, condition=condition) for ts in sw_times]) # get the phase at the time of each spike in the window to determine which are late-phase spikes
            late_mask = sw_phases >= phase_threshold # create mask to identify which spikes are late-phase based on the phase threshold
            late_ids = sw_ids[late_mask] # get IDs of the late-phase spikes that we will use for decoding  

            if late_ids.size < min_late_spikes: # if we don't have enough late-phase spikes to meet the minimum requirement for decoding, we skip this time point and record it as not decoded
                rows.append((t_eval, False, np.nan, np.nan, int(late_ids.size), pos_x, pos_y))
                continue

            pos_future = self._get_position_at_time(t_eval + future_window) # get position of agent at future time point that will serve as the true future heading for comparison
            if pos_now is None or pos_future is None: # if we can't get positions for current or future time, we can't decode direction, so we record this time point as not decoded and move on
                rows.append((t_eval, False, np.nan, np.nan, int(late_ids.size), pos_x, pos_y))
                continue

            true_vec = pos_future - pos_now # compute true future heading vector by taking the difference between true future and current position
            true_norm = np.linalg.norm(true_vec) # compute length of true future heading vector to check if valid for comparison -> if the agent didn't move (or moved very little), the true heading is not well-defined, so we skip decoding for this time point
            if true_norm < 1e-12: # if true future heading vector is too small, we cant meaningfully decode direction so we record this time point as not decoded and move on
                rows.append((t_eval, False, np.nan, np.nan, int(late_ids.size), pos_x, pos_y)) 
                continue
            true_dir = true_vec / true_norm # normalize true future heading to get unit vector for comparison with decoded direction

            decoded_dir = self._decode_direction_from_spikes(pos_now, late_ids) # decode heading direction from late-phase spikes using the population vector approach based on the centres of the active cells' place fields relative to the agent's current position
            if decoded_dir is None: # if decoding fails (e.g. no valid spikes or zero vector), we record this time point as not decoded and move on
                rows.append((t_eval, False, np.nan, np.nan, int(late_ids.size), pos_x, pos_y))
                continue
            
            # compute the dot product between decoded direction and true future direction to assess how well the decoded direction aligns with the actual future movement direction. We also compute the angle error in degrees for more intuitive interpretation of decoding accuracy. We record this time point as successfully decoded along with the computed metrics.
            dot = float(np.clip(np.dot(decoded_dir, true_dir), -1.0, 1.0)) # the dot product gives us a measure of how well the decoded direction aligns with the true future direction, where 1 means perfect alignment, 0 means orthogonal, and -1 means opposite directions. We clip the value to ensure numerical stability when computing the angle.
            angle_deg = float(np.degrees(np.arccos(dot))) # the angle error in degrees gives us an intuitive measure of decoding accuracy, where 0 degrees means perfect decoding and larger angles indicate worse decoding. We convert from radians to degrees for easier interpretation. 
            rows.append((t_eval, True, dot, angle_deg, int(late_ids.size), pos_x, pos_y)) # record this time point as successfully decoded along with the computed dot product, angle error, and number of late spikes used for decoding

        result_df = pd.DataFrame(rows, columns=['t', 'decoded', 'dot', 'angle_deg', 'n_late_spikes', 'pos_x', 'pos_y'])
        decoded_df = result_df[result_df['decoded'] == True]

        out = {
            'condition': condition,
            'n_eval_total': int(len(result_df)),
            'n_eval_decoded': int(len(decoded_df)),
            'coverage': float(len(decoded_df) / max(1, len(result_df))),
            'mean_angle_error_deg': float(decoded_df['angle_deg'].mean()) if len(decoded_df) else np.nan,
            'median_angle_error_deg': float(decoded_df['angle_deg'].median()) if len(decoded_df) else np.nan,
            'mean_directional_dot': float(decoded_df['dot'].mean()) if len(decoded_df) else np.nan,
            'phase_threshold': float(phase_threshold),
            't_window': float(t_window),
            'future_window': float(future_window),
            'sample_every': float(sample_every),
        }

        # Cache the full decode-evaluation timeseries so plotting can reuse
        # results without rerunning decoder analysis.
        if not hasattr(self, 'decoder_validation_cache'):
            self.decoder_validation_cache = {}
        self.decoder_validation_cache[condition] = {
            'timeseries': result_df.copy(),
            'phase_threshold': float(phase_threshold),
            't_window': float(t_window),
            'future_window': float(future_window),
            'sample_every': float(sample_every),
            'min_late_spikes': int(min_late_spikes),
        }

        if return_timeseries:
            out['timeseries'] = result_df
        return out

    def compare_decoder_conditions(self,
                                   phase_threshold=1.5 * np.pi,
                                   t_window=None,
                                   future_window=0.1,
                                   sample_every=0.05,
                                   min_late_spikes=1,
                                   return_timeseries=False):
        """Convenience wrapper to compare decode-only performance across all conditions."""
        conditions = ('theta', 'notheta', 'scrambled')
        summaries = []
        full = {}
        for cond in conditions:
            res = self.validate_late_phase_decoder(
                condition=cond,
                phase_threshold=phase_threshold,
                t_window=t_window,
                future_window=future_window,
                sample_every=sample_every,
                min_late_spikes=min_late_spikes,
                return_timeseries=return_timeseries,
            )
            full[cond] = res
            summaries.append({
                'condition': cond,
                'coverage': res.get('coverage', np.nan),
                'mean_angle_error_deg': res.get('mean_angle_error_deg', np.nan),
                'median_angle_error_deg': res.get('median_angle_error_deg', np.nan),
                'mean_directional_dot': res.get('mean_directional_dot', np.nan),
                'n_eval_total': res.get('n_eval_total', 0),
                'n_eval_decoded': res.get('n_eval_decoded', 0),
            })

        summary_df = pd.DataFrame(summaries)
        if return_timeseries:
            return summary_df, full
        return summary_df

    def validate_SR_based_decoder(self,
                                   which_W='W',
                                   future_window=0.1,
                                   sample_every=0.05,
                                   return_timeseries=False):
        """
        Decode direction using the learned SR matrix W itself, not spike patterns.
        
        This tests whether the learned SR structure (preferential weights to successor states)
        encodes useful directional information for navigation.
        
        Args:
            which_W: 'W', 'W_notheta', or 'W_scrambled' — which SR matrix to use
            future_window: Time offset (s) to compute true future heading
            sample_every: Temporal spacing (s) between decode evaluations
            return_timeseries: If True, include per-timepoint DataFrame
        
        Returns:
            dict with coverage, angular error stats, and directional correlation
        """
        # Get the requested W matrix
        if which_W == 'W':
            W = self.W
        elif which_W == 'W_notheta':
            W = self.W_notheta
        elif which_W == 'W_scrambled':
            W = self.W_scrambled
        else:
            return {'message': f'Unknown W matrix: {which_W}'}
        
        if W is None or W.size == 0:
            return {
                'which_W': which_W,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': f'{which_W} is empty or not initialized'
            }
        
        if self.history is None or len(self.history) < 3:
            return {
                'which_W': which_W,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': 'Not enough trajectory history'
            }
        
        t_min = float(self.history['t'].min())
        t_max = float(self.history['t'].max()) - future_window
        if t_max <= t_min:
            return {
                'which_W': which_W,
                'n_eval_total': 0,
                'n_eval_decoded': 0,
                'coverage': 0.0,
                'mean_angle_error_deg': np.nan,
                'median_angle_error_deg': np.nan,
                'mean_directional_dot': np.nan,
                'message': 'Time span too short for requested window'
            }
        
        eval_times = np.arange(t_min, t_max, sample_every)
        if eval_times.size == 0:
            eval_times = np.array([(t_min + t_max) / 2.0])
        
        rows = []
        for t_eval in eval_times:
            pos_now = self._get_position_at_time(t_eval)
            pos_future = self._get_position_at_time(t_eval + future_window)
            
            if pos_now is None or pos_future is None:
                rows.append((t_eval, False, np.nan, np.nan))
                continue
            
            true_vec = pos_future - pos_now
            true_norm = np.linalg.norm(true_vec)
            if true_norm < 1e-12:
                rows.append((t_eval, False, np.nan, np.nan))
                continue
            true_dir = true_vec / true_norm
            
            # Use a continuous gaussian state for SR decoding so we do not lose samples
            # when gaussianThreshold yields all-zeros between fields.
            state = self.posToState(pos_now, stateType='gaussianThreshold', normalise=False)
            if state is None:
                rows.append((t_eval, False, np.nan, np.nan))
                continue
            state = np.asarray(state, dtype=float).reshape(-1)
            if state.size != self.nCells:
                rows.append((t_eval, False, np.nan, np.nan))
                continue
            state_sum = np.sum(state)
            if state_sum < 1e-12:
                # Fallback: use nearest place cell when activity is numerically tiny.
                nearest = int(np.argmin(np.linalg.norm(self.centres - pos_now[np.newaxis, :], axis=1)))
                state = np.zeros(self.nCells, dtype=float)
                state[nearest] = 1.0
            
            # STDP update uses W[post, pre], so for a current cell (pre)
            # the future distribution is in the corresponding COLUMN.
            active_cells = np.where(state > 0)[0]
            if len(active_cells) == 0:
                rows.append((t_eval, False, np.nan, np.nan))
                continue
            
            # Weight successors by state activity
            weighted_successors = np.zeros(self.nCells)
            for cell_id in active_cells:
                weighted_successors += state[cell_id] * W[:, cell_id]
            
            # Decode direction as weighted vector to successor cell centres
            successors = np.where(weighted_successors > 1e-12)[0]
            if len(successors) == 0:
                rows.append((t_eval, False, np.nan, np.nan))
                continue
            
            vectors = self.centres[successors] - pos_now[np.newaxis, :]
            weights = weighted_successors[successors]
            weighted_vec = np.sum(vectors * weights[:, np.newaxis], axis=0)
            
            norm = np.linalg.norm(weighted_vec)
            if norm < 1e-12:
                rows.append((t_eval, False, np.nan, np.nan))
                continue
            
            decoded_dir = weighted_vec / norm
            dot = float(np.clip(np.dot(decoded_dir, true_dir), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(dot)))
            rows.append((t_eval, True, dot, angle_deg))
        
        result_df = pd.DataFrame(rows, columns=['t', 'decoded', 'dot', 'angle_deg'])
        decoded_df = result_df[result_df['decoded'] == True]
        
        out = {
            'which_W': which_W,
            'n_eval_total': int(len(result_df)),
            'n_eval_decoded': int(len(decoded_df)),
            'coverage': float(len(decoded_df) / max(1, len(result_df))),
            'mean_angle_error_deg': float(decoded_df['angle_deg'].mean()) if len(decoded_df) else np.nan,
            'median_angle_error_deg': float(decoded_df['angle_deg'].median()) if len(decoded_df) else np.nan,
            'mean_directional_dot': float(decoded_df['dot'].mean()) if len(decoded_df) else np.nan,
            'future_window': float(future_window),
            'sample_every': float(sample_every),
        }
        
        if return_timeseries:
            out['timeseries'] = result_df
        return out

    def compare_SR_and_spike_decoders(self,
                                       phase_threshold=1.5 * np.pi,
                                       t_window=None,
                                       future_window=0.1,
                                       sample_every=0.05,
                                       min_late_spikes=1):
        """
        Compare spike-based and SR-based decoders side-by-side.
        
        Returns a comparison table showing whether spikes or SR better predict future direction.
        """
        if t_window is None:
            t_window = 2 / self.thetaFreq
        
        spike_results = self.compare_decoder_conditions(
            phase_threshold=phase_threshold,
            t_window=t_window,
            future_window=future_window,
            sample_every=sample_every,
            min_late_spikes=min_late_spikes,
            return_timeseries=False
        )
        
        sr_results_W = self.validate_SR_based_decoder(
            which_W='W',
            future_window=future_window,
            sample_every=sample_every,
            return_timeseries=False
        )
        
        sr_results_W_notheta = self.validate_SR_based_decoder(
            which_W='W_notheta',
            future_window=future_window,
            sample_every=sample_every,
            return_timeseries=False
        )
        
        sr_results_W_scrambled = self.validate_SR_based_decoder(
            which_W='W_scrambled',
            future_window=future_window,
            sample_every=sample_every,
            return_timeseries=False
        )
        
        # Build comparison table
        comparison = pd.DataFrame([
            {'decoder': 'spike (theta)', 'mean_directional_dot': spike_results.iloc[0]['mean_directional_dot'], 'mean_angle_error_deg': spike_results.iloc[0]['mean_angle_error_deg'], 'coverage': spike_results.iloc[0]['coverage']},
            {'decoder': 'spike (notheta)', 'mean_directional_dot': spike_results.iloc[1]['mean_directional_dot'], 'mean_angle_error_deg': spike_results.iloc[1]['mean_angle_error_deg'], 'coverage': spike_results.iloc[1]['coverage']},
            {'decoder': 'spike (scrambled)', 'mean_directional_dot': spike_results.iloc[2]['mean_directional_dot'], 'mean_angle_error_deg': spike_results.iloc[2]['mean_angle_error_deg'], 'coverage': spike_results.iloc[2]['coverage']},
            {'decoder': 'SR (W)', 'mean_directional_dot': sr_results_W['mean_directional_dot'], 'mean_angle_error_deg': sr_results_W['mean_angle_error_deg'], 'coverage': sr_results_W['coverage']},
            {'decoder': 'SR (W_notheta)', 'mean_directional_dot': sr_results_W_notheta['mean_directional_dot'], 'mean_angle_error_deg': sr_results_W_notheta['mean_angle_error_deg'], 'coverage': sr_results_W_notheta['coverage']},
            {'decoder': 'SR (W_scrambled)', 'mean_directional_dot': sr_results_W_scrambled['mean_directional_dot'], 'mean_angle_error_deg': sr_results_W_scrambled['mean_angle_error_deg'], 'coverage': sr_results_W_scrambled['coverage']},
        ])
        
        return comparison

    def evaluate_junction_choices(self,
                                  decoder_type='spike',
                                  condition='theta',
                                  which_W='W',
                                  phase_threshold=1.5 * np.pi,
                                  t_window=None,
                                  sample_every=0.05,
                                  min_late_spikes=1,
                                  junction_x=None,
                                  x_band=None,
                                  y_center=None,
                                  y_band=None,
                                  future_choice_window=0.4,
                                  decision_gap=0.3,
                                  return_timeseries=False):
        """
        Evaluate left/right decision accuracy at the T-maze junction.

        A decision event is sampled near the junction and classified by the sign of
        y movement after `future_choice_window` seconds:
            y_future > y_now -> left arm
            y_future < y_now -> right arm

        Decoded choice is obtained from either:
            - spike decoder: late-phase CA1 spikes (condition-specific)
            - SR decoder: W-based directional readout
        """
        if self.mazeType != 'TMaze':
            return {
                'message': 'Junction decision evaluation currently implemented for TMaze only.',
                'decoder_type': decoder_type,
                'n_events_total': 0,
                'n_events_decoded': 0,
                'coverage': 0.0,
                'choice_accuracy': np.nan,
            }

        if self.history is None or len(self.history) < 3:
            return {
                'message': 'Not enough trajectory history.',
                'decoder_type': decoder_type,
                'n_events_total': 0,
                'n_events_decoded': 0,
                'coverage': 0.0,
                'choice_accuracy': np.nan,
            }

        if t_window is None:
            t_window = 2 / self.thetaFreq
        if junction_x is None:
            junction_x = float(self.roomSize)
        if x_band is None:
            x_band = 0.06 * float(self.roomSize)
        if y_center is None:
            y_center = float(self.roomSize)
        if y_band is None:
            y_band = 0.12 * float(self.roomSize)

        hist = self.history
        t_min = float(hist['t'].min()) + t_window
        t_max = float(hist['t'].max()) - future_choice_window
        if t_max <= t_min:
            return {
                'message': 'Time span too short for junction evaluation windows.',
                'decoder_type': decoder_type,
                'n_events_total': 0,
                'n_events_decoded': 0,
                'coverage': 0.0,
                'choice_accuracy': np.nan,
            }

        eval_times = np.arange(t_min, t_max, sample_every)
        if eval_times.size == 0:
            eval_times = np.array([(t_min + t_max) / 2.0])

        # Keep only times where position is at the junction corridor.
        candidate_times = []
        for t_eval in eval_times:
            pos_now = self._get_position_at_time(t_eval)
            if pos_now is None:
                continue
            if (abs(pos_now[0] - junction_x) <= x_band) and (abs(pos_now[1] - y_center) <= y_band):
                candidate_times.append(float(t_eval))

        # Thin near-duplicate samples around the same visit to one decision event.
        event_times = []
        last_t = -np.inf
        for t_eval in candidate_times:
            if t_eval - last_t >= decision_gap:
                event_times.append(t_eval)
                last_t = t_eval

        # Prepare data needed by spike decoder.
        if decoder_type == 'spike':
            if not hasattr(self, 'spikedata_by_condition'):
                return {
                    'message': 'No condition-specific spike data found.',
                    'decoder_type': decoder_type,
                    'n_events_total': len(event_times),
                    'n_events_decoded': 0,
                    'coverage': 0.0,
                    'choice_accuracy': np.nan,
                }
            if condition not in self.spikedata_by_condition:
                return {
                    'message': f'Condition {condition} not found in spike data.',
                    'decoder_type': decoder_type,
                    'n_events_total': len(event_times),
                    'n_events_decoded': 0,
                    'coverage': 0.0,
                    'choice_accuracy': np.nan,
                }
            spike_dict = self.spikedata_by_condition[condition].get('CA1', {'times': [], 'ids': []})
            spike_times = np.asarray(spike_dict.get('times', []), dtype=float).reshape(-1)
            spike_ids = np.asarray(spike_dict.get('ids', []), dtype=int).reshape(-1)

        elif decoder_type == 'sr':
            if which_W == 'W':
                W = self.W
            elif which_W == 'W_notheta':
                W = self.W_notheta
            elif which_W == 'W_scrambled':
                W = self.W_scrambled
            else:
                return {
                    'message': f'Unknown W matrix: {which_W}',
                    'decoder_type': decoder_type,
                    'n_events_total': len(event_times),
                    'n_events_decoded': 0,
                    'coverage': 0.0,
                    'choice_accuracy': np.nan,
                }
            if W is None or W.size == 0:
                return {
                    'message': f'{which_W} is empty or not initialized.',
                    'decoder_type': decoder_type,
                    'n_events_total': len(event_times),
                    'n_events_decoded': 0,
                    'coverage': 0.0,
                    'choice_accuracy': np.nan,
                }
        else:
            return {
                'message': "decoder_type must be 'spike' or 'sr'.",
                'decoder_type': decoder_type,
                'n_events_total': len(event_times),
                'n_events_decoded': 0,
                'coverage': 0.0,
                'choice_accuracy': np.nan,
            }

        rows = []
        for t_eval in event_times:
            pos_now = self._get_position_at_time(t_eval)
            pos_future = self._get_position_at_time(t_eval + future_choice_window)
            if pos_now is None or pos_future is None:
                rows.append((t_eval, False, np.nan, np.nan, np.nan, np.nan))
                continue

            dy_true = float(pos_future[1] - pos_now[1])
            if abs(dy_true) < 1e-9:
                rows.append((t_eval, False, np.nan, np.nan, np.nan, dy_true))
                continue
            true_choice = 1 if dy_true > 0 else -1  # +1 left, -1 right

            decoded_dir = None
            if decoder_type == 'spike':
                t0 = t_eval - t_window
                t1 = t_eval
                in_window = (spike_times >= t0) & (spike_times <= t1)
                if np.any(in_window):
                    sw_times = spike_times[in_window]
                    sw_ids = spike_ids[in_window]
                    sw_phases = np.array([self.get_phase_at_time(ts, condition=condition) for ts in sw_times])
                    late_mask = sw_phases >= phase_threshold
                    late_ids = sw_ids[late_mask]
                    if late_ids.size >= min_late_spikes:
                        decoded_dir = self._decode_direction_from_spikes(pos_now, late_ids)
            else:
                state = self.posToState(pos_now, stateType='gaussianThreshold', normalise=False)
                if state is not None:
                    state = np.asarray(state, dtype=float).reshape(-1)
                    if state.size == self.nCells:
                        if np.sum(state) < 1e-12:
                            nearest = int(np.argmin(np.linalg.norm(self.centres - pos_now[np.newaxis, :], axis=1)))
                            state = np.zeros(self.nCells, dtype=float)
                            state[nearest] = 1.0
                        active_cells = np.where(state > 0)[0]
                        if len(active_cells) > 0:
                            weighted_successors = np.zeros(self.nCells)
                            for cell_id in active_cells:
                                weighted_successors += state[cell_id] * W[:, cell_id]
                            successors = np.where(weighted_successors > 1e-12)[0]
                            if len(successors) > 0:
                                vectors = self.centres[successors] - pos_now[np.newaxis, :]
                                weights = weighted_successors[successors]
                                weighted_vec = np.sum(vectors * weights[:, np.newaxis], axis=0)
                                norm = np.linalg.norm(weighted_vec)
                                if norm > 1e-12:
                                    decoded_dir = weighted_vec / norm

            if decoded_dir is None:
                rows.append((t_eval, False, np.nan, np.nan, true_choice, dy_true))
                continue

            dy_dec = float(decoded_dir[1])
            if abs(dy_dec) < 1e-12:
                rows.append((t_eval, False, np.nan, dy_dec, true_choice, dy_true))
                continue
            decoded_choice = 1 if dy_dec > 0 else -1
            correct = float(decoded_choice == true_choice)
            rows.append((t_eval, True, correct, dy_dec, true_choice, dy_true))

        result_df = pd.DataFrame(
            rows,
            columns=['t', 'decoded', 'correct_choice', 'decoded_dy', 'true_choice', 'true_dy']
        )
        decoded_df = result_df[result_df['decoded'] == True]

        out = {
            'decoder_type': decoder_type,
            'condition': condition if decoder_type == 'spike' else None,
            'which_W': which_W if decoder_type == 'sr' else None,
            'n_events_total': int(len(result_df)),
            'n_events_decoded': int(len(decoded_df)),
            'coverage': float(len(decoded_df) / max(1, len(result_df))),
            'choice_accuracy': float(decoded_df['correct_choice'].mean()) if len(decoded_df) else np.nan,
            'junction_x': float(junction_x),
            'future_choice_window': float(future_choice_window),
        }
        if return_timeseries:
            out['timeseries'] = result_df
        return out


## Non-class functions    
        
def getWalls(mazeType, roomSize=1):
    """Stores and returns dictionaries containing all the walls of a maze
    Args:
        mazeType (str): Name of the maze 
        roomSize (int, optional): scaling parameter for roomsize. Defaults to 1 metre.
    Returns:
        dict: wall dictionary
    """    
    walls = {}
    rs = roomSize
    if mazeType == 'oneRoom':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0]],
                                [[rs,0],[0,0]]])
    elif mazeType == 'twoRooms':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0.6*rs]],
                                [[rs,0.4*rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[rs,0.4*rs]],
                                [[rs,0.6*rs],[rs,rs]],
                                [[rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,0]],
                                [[2*rs,0],[rs,0]]])
        walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]]])
    elif mazeType == 'fourRooms':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[0.4*rs,rs]],
                                [[0.6*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,0.6*rs]],
                                [[rs,0.4*rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[rs,0.4*rs]],
                                [[rs,0.6*rs],[rs,rs]],
                                [[rs,rs],[1.4*rs,rs]],
                                [[1.6*rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,0]],
                                [[2*rs,0],[rs,0]]])
        walls['room3'] = np.array([
                                [[0,rs],[0.4*rs,rs]],
                                [[0.6*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,1.4*rs]],
                                [[rs,1.6*rs],[rs,2*rs]],
                                [[rs,2*rs],[0,2*rs]],
                                [[0,2*rs],[0,rs]]])
        walls['room4'] = np.array([
                                [[rs,rs],[1.4*rs,rs]],
                                [[1.6*rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,2*rs]],
                                [[2*rs,2*rs],[rs,2*rs]],
                                [[rs,2*rs],[rs,1.6*rs]],
                                [[rs,1.4*rs],[rs,rs]]])
        walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]],
                                        [[0.4*rs,rs],[0.6*rs,rs]],
                                        [[rs,1.4*rs],[rs,1.6*rs]],
                                        [[1.4*rs,rs],[1.6*rs,rs]]])
    elif mazeType == 'twoRoomPassage':
        walls['room1'] = np.array([
                                [[0,0],[rs,0]],
                                [[rs,0],[rs,rs]],
                                [[rs,rs],[0.75*rs,rs]],
                                [[0.25*rs,rs],[0,rs]],
                                [[0,rs],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[2*rs,0]],
                                [[2*rs,0],[2*rs,rs]],
                                [[2*rs,rs],[1.75*rs,rs]],
                                [[1.25*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,0]]])
        walls['room3'] = np.array([
                                [[0,rs],[0,1.4*rs]],
                                [[0,1.4*rs],[2*rs,1.4*rs]],
                                [[2*rs,1.4*rs],[2*rs,rs]]])
        walls['doors'] = np.array([[[0.25*rs,rs],[0.75*rs,rs]],
                                [[1.25*rs,rs],[1.75*rs,rs]]])
    elif mazeType == 'longCorridor':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['longbarrier'] = np.array([
                                [[0.1*rs,0],[0.1*rs,0.9*rs]],
                                [[0.2*rs,rs],[0.2*rs,0.1*rs]],
                                [[0.3*rs,0],[0.3*rs,0.9*rs]],
                                [[0.4*rs,rs],[0.4*rs,0.1*rs]],
                                [[0.5*rs,0],[0.5*rs,0.9*rs]],
                                [[0.6*rs,rs],[0.6*rs,0.1*rs]],
                                [[0.7*rs,0],[0.7*rs,0.9*rs]],
                                [[0.8*rs,rs],[0.8*rs,0.1*rs]],
                                [[0.9*rs,0],[0.9*rs,0.9*rs]]])
    elif mazeType == 'rectangleRoom':
        ratio = np.pi/2.8
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[ratio*rs,rs]],
                                [[ratio*rs,rs],[ratio*rs,0]],
                                [[ratio*rs,0],[0,0]]])
    elif mazeType == 'loop':
        height = 0.2
        walls['room'] = np.array([
                                [[0,0],[rs,0]],
                                [[0,height],[rs,height]]])
        walls['doors'] = np.array([
                                [[0,0],[0,height]],
                                [[rs,0],[rs,height]]])
    
    elif mazeType == 'TMaze':
        corridor_half_width = 0.05 * rs
        arm_width = 2 * corridor_half_width
        walls['corridors'] = np.array([
                                [[0,rs + corridor_half_width],[rs,rs + corridor_half_width]], #  
                                [[0,rs - corridor_half_width],[rs,rs - corridor_half_width]],  
                                [[rs,rs + corridor_half_width],[rs,2 * rs]], 
                                [[rs + arm_width,2 * rs],[rs + arm_width,0]],  
                                [[rs,rs - corridor_half_width],[rs,0]]]) # 
        walls['doors'] = np.array([[[rs,rs - corridor_half_width - 0.5 * arm_width],[rs + arm_width,rs - corridor_half_width - 0.5 * arm_width]]])                               

    return walls

#MOVEMENT FUNCTIONS
def wallBounceOrFollow(currentDirection,wall,whatToDo='bounce'):
    """Given current direction, and wall and an instruction returns a new direction which is the result of implementing that instruction on the current direction
        wrt the wall. e.g. 'bounce' returns direction after elastic bounce off wall. 'follow' returns direction parallel to wall (closest to current heading)
    Args:
        currentDirection (array): the current direction vector
        wall (array): start and end coordinates of the wall
        whatToDo (str, optional): 'bounce' or 'follow'. Defaults to 'bounce'.
    Returns:
        array: new direction
    """    
    if whatToDo == 'bounce':
        wallPerp = perp(wall[1] - wall[0]) # perp() is a function that returns the perpendicular of a vector. wall[1] is the end of the wall, wall[0] is the start, so this gives the outward normal to the wall (pointing to the right of the direction from start to end)
        if np.dot(wallPerp,currentDirection) <= 0: # 
            wallPerp = -wallPerp #it is now the perpendicular with smallest angle to dir 
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,currentDirection) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar, wallPerp = wallPar/np.linalg.norm(wallPar), wallPerp/np.linalg.norm(wallPerp) #normalise
        dir_ = wallPar * np.dot(currentDirection,wallPar) - wallPerp * np.dot(currentDirection,wallPerp)
        newDir = dir_/np.linalg.norm(dir_)
    elif whatToDo == 'follow':
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,currentDirection) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar = wallPar/np.linalg.norm(wallPar)
        dir_ = wallPar * np.dot(currentDirection,wallPar)
        # DEBUG: if dir_ is non finite or nan, print out the values
        if not np.isfinite(dir_).all():
            print(f"Non-finite dir_ in follow: dir_={dir_}, wallPar={wallPar}, currentDirection={currentDirection}, dot={np.dot(currentDirection, wallPar)}")
        newDir = dir_/np.linalg.norm(dir_)
    return newDir

def turn(currentDirection, turnAngle):
    """Turns the current direction by an amount theta, modulus 2pi
    Args:
        currentDirection (array): current direction 2-vector
        turnAngle (float): angle ot turn in radians
    Returns:
        array: new direction
    """    
    theta_ = theta(currentDirection)
    theta_ += turnAngle
    theta_ = np.mod(theta_, 2*np.pi)
    newDirection = np.array([np.cos(theta_),np.sin(theta_)])
    return newDirection

def perp(a=None):
    """Given 2-vector, a, returns its perpendicular
    Args:
        a (array, optional): 2-vector direction. Defaults to None.
    Returns:
        array: perpendicular to a
    """    
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0] 
    return b

def theta(segment):
    """Given a 'segment' (either 2x2 start and end positions or 2x1 direction bearing) 
         returns the 'angle' of this segment modulo 2pi
    Args:
        segment (array): The segment, (2,2) or (2,) array 
    Returns:
        float: angle of segment
    """    
    eps = 1e-6
    if segment.shape == (2,): 
        return np.mod(np.arctan2(segment[1],(segment[0] + eps)),2*np.pi)
    elif segment.shape == (2,2):
        return np.mod(np.arctan2((segment[1][1]-segment[0][1]),(segment[1][0] - segment[0][0] + eps)), 2*np.pi)


## Visualisation class and functions

class Visualiser():
    def __init__(self, mazeAgent):
        self.mazeAgent = mazeAgent
        self.snapshots = mazeAgent.snapshots
        self.history = mazeAgent.history

    def plotMazeStructure(self,fig=None,ax=None,hist_id=-1,save=False):
        snapshot = self.snapshots.iloc[hist_id]
        extent, walls = snapshot['mazeState']['extent'], snapshot['mazeState']['walls']
        if (fig, ax) == (None, None): 
            fig, ax = plt.subplots(figsize=(4*(extent[1]-extent[0]),4*(extent[3]-extent[2]))) # 4* in Tom's code instead of 1*  
            #fig, ax = plt.subplots(figsize=(4.5,0.5)) # for loop
            #fig, ax = plt.subplots(figsize=(3,4)) #3,4 for value function, 1,2 for place field reconstructions
        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=1.0) # was 10 in Tom's code
            ax.set_xlim(left=extent[0]-0.05,right=extent[1]+0.3)
            ax.set_ylim(bottom=extent[2]-0.05,top=extent[3]+0.05)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        if save == True: 
            saveFigure(fig, 'mazeStructure')
        return fig, ax
    
    def plotTrajectory(self,fig=None, ax=None, hist_id=-1,starttime=0,endtime=2,plotAsLine=False, save=False):
        skiprate = max(1,int(0.015/(self.mazeAgent.speedScale * self.mazeAgent.dt)))
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id) # hist_id is used to get the maze structure, but starttime and endtime are used to get the trajectory, so they can be different
        startid = self.history['t'].sub(starttime*60).abs().to_numpy().argmin() # convert starttime and endtime from minutes to seconds, then find closest time in history and get index
        endid = self.history['t'].sub(endtime*60).abs().to_numpy().argmin() # convert starttime and endtime from minutes to seconds, then find closest time in history and get index
        trajectory = np.stack(self.history['pos'][startid:endid])[::skiprate] # get positions between startid and endid, then skip some to avoid overcrowding the plot
        if plotAsLine == False:
            ax.scatter(trajectory[:,0],trajectory[:,1],s=10,alpha=0.7,zorder=2)
        elif plotAsLine == True:
            ax.plot(trajectory[:,0],trajectory[:,1])
        if save == True:
            saveFigure(fig, "trajectory")
        return fig, ax
    
    def plotEpTrajectory(self, fig=None, ax=None, hist=None, hist_id=-1, starttime=0.0, endtime=2.0, plotAsLine=False, save=False, color='#ff5ebb'):
        skiprate = max(1, int(0.015 / (self.mazeAgent.speedScale * self.mazeAgent.dt)))
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        hist_df = hist if hist is not None else self.history
        t = np.asarray(hist_df['t'].values, dtype=float)
        starttime = float(starttime)
        endtime = float(endtime)
        if endtime <= starttime:
            raise ValueError(f"endtime ({endtime}) must be > starttime ({starttime})")
        startid = np.searchsorted(t, starttime, side='left')
        endid = np.searchsorted(t, endtime, side='right')
        if startid >= endid:
            raise ValueError(f"No trajectory data in [{starttime}, {endtime}] (history t range {t.min()}–{t.max()})")
        pos_list = hist_df['pos'].iloc[startid:endid].tolist()
        trajectory = np.vstack(pos_list)[::skiprate]
        if not plotAsLine:
            ax.scatter(trajectory[:,0], trajectory[:,1], s=4, alpha=0.3, zorder=2, color=color, marker='.', edgecolors='none', lw=0)
        else:
            ax.plot(trajectory[:,0], trajectory[:,1])
        return fig, ax

    def animateEpTrajectory(self, fig=None, ax=None, hist=None, hist_id=-1,
                            starttime=0.0, endtime=2.0, speedup=4.0, fps=20,
                            trail_length=None, save_path='trajectory.gif',
                            color='#ff5ebb', agentSize=0.15, trailAlpha=0.3,
                            showGoal=True, dpi=100):
        """
        Animate the agent's trajectory between starttime and endtime, saved as a
        sped-up GIF. Draws the agent's heading as a wedge, plus a per-frame
        arrow showing the decoded direction (falling back to the actual motion
        direction when no decoded direction is available for that frame).

        NOTE: frames are rendered and assembled manually with Pillow (rather than
        via matplotlib.animation + PillowWriter), because PillowWriter's GIF
        output doesn't set a per-frame disposal method. Without it, GIF decoders
        only paint the *changed* region of each frame on top of the previous
        one instead of clearing first, so moving artists (the wedge, the arrow)
        visually "smear"/accumulate across frames. Writing the GIF directly with
        disposal=2 forces each frame to be fully cleared before the next is
        drawn, which fixes it.

        Parameters
        ----------
        speedup : float
            Playback speed relative to real simulation time. speedup=4 means
            4 seconds of simulated time play out in 1 second of GIF.
        fps : int
            Frames per second of the output GIF.
        trail_length : int or None
            Number of past samples to show as a fading trail behind the agent.
            None keeps the full accumulated trail (like the mp4 version).
        agentSize : float
            Radius of the agent wedge marker.
        trailAlpha : float
            Alpha of the trail scatter dots.
        showGoal : bool
            Whether to plot the goal location on the maze.
        save_path : str
            Output path for the GIF.
        """
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
            ax.set_aspect('equal')

        if dpi is not None:
            fig.set_dpi(dpi)

        if showGoal:
            self.plotGoal(fig=fig, ax=ax)

        hist_df = hist if hist is not None else self.history
        t = np.asarray(hist_df['t'].values, dtype=float)

        starttime = float(starttime)
        endtime = float(endtime)
        if endtime <= starttime:
            raise ValueError(f"endtime ({endtime}) must be > starttime ({starttime})")

        startid = np.searchsorted(t, starttime, side='left')
        endid = np.searchsorted(t, endtime, side='right')
        if startid >= endid:
            raise ValueError(
                f"No trajectory data in [{starttime}, {endtime}] "
                f"(history t range {t.min()}-{t.max()})"
            )

        dt = self.mazeAgent.dt  # real time between consecutive raw samples

        # How much simulated time should pass per rendered GIF frame
        sim_time_per_frame = speedup / fps
        frame_skiprate = max(1, int(round(sim_time_per_frame / dt)))

        pos_list = hist_df['pos'].iloc[startid:endid].tolist()
        trajectory = np.vstack(pos_list)[::frame_skiprate]

        # decoded directions over the same window, downsampled to match trajectory
        decoded_dirs_raw = self.mazeAgent.decoded_dir_history[startid:endid]
        if len(decoded_dirs_raw) == 0:
            decoded_dirs = np.zeros((0, 2))
        else:
            decoded_dirs = np.array(decoded_dirs_raw)[::frame_skiprate]
            if decoded_dirs.ndim == 1:
                decoded_dirs = decoded_dirs.reshape(-1, 2)

        if len(trajectory) < 2:
            raise ValueError(
                "Not enough samples to animate — lower `speedup`, raise `fps`, "
                "or widen [starttime, endtime]."
            )

        # fallback direction = actual motion direction, used wherever decoded dir is missing/NaN
        motion_dirs = np.zeros_like(trajectory)
        motion_dirs[:-1] = trajectory[1:] - trajectory[:-1]
        motion_dirs[-1] = motion_dirs[-2] if len(motion_dirs) > 1 else np.array([1.0, 0.0])
        motion_norms = np.linalg.norm(motion_dirs, axis=1, keepdims=True)
        motion_norms[motion_norms < 1e-12] = 1e-12
        motion_dirs = motion_dirs / motion_norms

        valid_mask = np.zeros(len(trajectory), dtype=bool)
        if len(decoded_dirs) > 0:
            valid_count = min(len(decoded_dirs), len(valid_mask))
            valid_mask[:valid_count] = ~np.any(np.isnan(decoded_dirs[:valid_count]), axis=1)

        # --- artists (created once, mutated in place each frame) ---
        trail_scatter = ax.scatter([], [], s=4, alpha=trailAlpha, zorder=2, color=color,
                                    marker='.', edgecolors='none', lw=0)
        agent_wedge = patches.Wedge((trajectory[0, 0], trajectory[0, 1]), agentSize, 0, 60,
                                     ec='red', fc='red', alpha=0.8, zorder=3)
        ax.add_patch(agent_wedge)
        quiver = ax.quiver([trajectory[0, 0]], [trajectory[0, 1]], [1], [0],
                            angles='xy', scale_units='xy', scale=1, color='C1',
                            width=0.005, zorder=4)

        def heading_for(i):
            if i < len(trajectory) - 1:
                direction = trajectory[i + 1] - trajectory[i]
            elif i > 0:
                direction = trajectory[i] - trajectory[i - 1]
            else:
                direction = np.array([1.0, 0.0])
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 1e-6 else np.array([1.0, 0.0])

        frames = []
        for i in range(len(trajectory)):
            pos = trajectory[i]

            # fading or accumulating trail
            trail_start = 0 if trail_length is None else max(0, i - trail_length)
            trail_scatter.set_offsets(trajectory[trail_start:i + 1])

            # wedge oriented along direction of motion
            direction = heading_for(i)
            heading_angle = np.degrees(np.arctan2(direction[1], direction[0])) + 180
            agent_wedge.set_center(pos)
            agent_wedge.set_theta1(heading_angle - 30)
            agent_wedge.set_theta2(heading_angle + 30)

            # decoded direction arrow for THIS frame only, falls back to motion direction
            arrow_dir = decoded_dirs[i] if valid_mask[i] else motion_dirs[i]
            quiver.set_offsets([pos])
            quiver.set_UVC(arrow_dir[0], arrow_dir[1])

            # render the full canvas and capture it as a standalone frame
            fig.canvas.draw()
            frame_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(Image.fromarray(frame_rgb, 'RGB'))

        # disposal=2 forces each frame to be fully cleared before the next is drawn,
        # which is what actually fixes the smearing/accumulation
        frames[0].save(
            save_path, save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0, disposal=2, optimize=False
        )
        plt.close(fig)
        print(f"Saved animation to {save_path}")
        return save_path

    
    def plotM(self,hist_id=-1, time=None, M=None,fig=None,ax=None,save=True,savename="",title="",show=True,plotTimeStamp=False,colorbar=True,whichM='M',colormatchto='TD_M'):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
        snapshot = self.snapshots.iloc[hist_id]
        if (ax is not None) and (fig is not None): 
            ax.clear()
        else:
            fig, ax = plt.subplots(figsize=(1,1))
        if M is None:
            if whichM == 'M': M = snapshot['M'].copy()
            elif whichM == 'W': M = snapshot['W'].copy()
            elif whichM == 'M_theta': M = self.mazeAgent.M_theta.copy()
            # elif whichM == 'W_notheta': M = self.mazeAgent.W_notheta.copy() by Livi to be consistent with plotPlaceField()
            elif whichM == 'W_notheta': M = snapshot['W_notheta'].copy() # by Livi to be consistent with plotPlaceField()
            # elif whichM == 'W_scrambled': M = self.mazeAgent.W_scrambled.copy() # added/#ed by Livi
            elif whichM == 'W_scrambled': M = snapshot['W_scrambled'].copy() # added/#ed by Livi

        t = int(np.round(snapshot['t']))
        most_positive = np.max(M)
        most_negative = np.min(M)

        if colormatchto == 'TD_M': 
            M_colormatch = self.mazeAgent.M
        elif colormatchto == 'W_onDiag': 
            M_colormatch = self.mazeAgent.W_onDiag
        # elif colormatchto is not None:
        #     M_colormatch = np.load(colormatchto)
        else: 
            M_colormatch = np.array([-1,1])

        # if np.min(M)/np.min(M_colormatch) > np.max(M)/np.max(M_colormatch):
            # M *= np.min(M_colormatch)/np.min(M)
        # else:
        M_ = M.copy()
        # np.fill_diagonal(M_,0)
        non_diag_max = np.max(M_)
        non_diag_kind_max = np.mean(M_[M_>0.9*non_diag_max])
        M *= np.max(M_colormatch)/non_diag_kind_max

        im = ax.imshow(M,cmap='viridis',vmin=np.min(M_colormatch),vmax=np.max(M_colormatch))
        divider = make_axes_locatable(ax)
        try: cax.clear()
        except: 
            pass
        if colorbar == True:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(im, cax=cax, ticks=[0])
            cb.outline.set_visible(False)

        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        ax.set_title(title)
        if save==True:
            saveFigure(fig, "M"+savename)
        if plotTimeStamp == True: 
            ax.text(100, 5,"%g s"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
        if show==False:
            plt.close(fig)
        
        try:
            return fig, ax, cb, cax
        except:
            return fig, ax 

    def addTimestamp(self, fig, ax, i=-1):
        t = self.mazeAgent.saveHist[i]['t']
        ax.text(x=0, y=0, t="%.2f" %t)

    def plotPlaceField(self, M=None, hist_id=-1, time=None, fig=None, ax=None, number=None, show=True, animationCall=False, plotTimeStamp=False,save=True,STDP=False,threshold=None,fitEllipse_=False,no_theta=False, scrambled = False):
        if M is None: 
            #time in minutes
            if time is not None: 
                hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
            #if a figure/ax objects are passed, clear the axis and replot the maze
            if (ax is not None) and (fig is not None): 
                ax.clear()
                self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
            # else if they are not passed plot the maze
            if (fig, ax) == (None, None):
                fig, ax = self.plotMazeStructure(hist_id=hist_id)
            
            if number == None: number = np.random.randint(0,self.mazeAgent.stateSize-1)
            
            snapshot = self.snapshots.iloc[hist_id]
            M = snapshot['M']
            if STDP==True: 
                if no_theta == True:
                    #snapshot = self.snapshots.iloc[hist_id-22] by Livi to take same snapshot for all conditions
                    M = snapshot['W_notheta']
                elif scrambled == True:
                    #snapshot = self.snapshots.iloc[hist_id-22] by Livi
                    M = snapshot['W_scrambled']
                else:
                    M = snapshot['W']
            t = int(np.round(snapshot['t'] / 60))
        else:
            snapshot = self.snapshots.iloc[hist_id]
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        extent = snapshot['mazeState']['extent']
        placeFields = self.mazeAgent.getPlaceFields(M=M,threshold=threshold)
        ax.imshow(placeFields[number],extent=extent,interpolation=None)

        if fitEllipse_ == True: 
            (X,Y,Z),_,_ = fitEllipse(placeFields[number],coords=self.mazeAgent.discreteCoords)
            ax.contour(X, Y, Z, levels=[1], colors=('w'), linewidths=2, linestyles="dashed")

        
        if self.mazeAgent.mazeType == 'loop': 
            x = self.mazeAgent.discreteCoords[10,:,0] 

        peakid= np.argmax(placeFields[number])
        peakcoord = self.mazeAgent.discreteCoords.reshape(-1,2)[peakid]
        ax.scatter(peakcoord[0],peakcoord[1],marker='x',s=130,color='darkgrey',linewidth=4,alpha=1)

        if plotTimeStamp == True: 
            ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
        if show==False:
            plt.close(fig)
        if save==True:
            saveFigure(fig, "placeField")
        return fig, ax
    

    def plotReceptiveField(self, number=None, hist_id=-1, fig=None, ax=None, show=True, fitEllipse_=False, save=False):
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        if number == None: number = np.random.randint(0,self.mazeAgent.nCells-1)
        extent = self.mazeAgent.extent
        rf = self.mazeAgent.discreteStates[..., number]
        ax.imshow(rf,extent=extent,interpolation=None)
        peakid= np.argmax(rf)
        peakcoord = self.mazeAgent.discreteCoords.reshape(-1,2)[peakid]
        ax.scatter(peakcoord[0],peakcoord[1],marker='x',s=100,color='darkgrey',linewidth=2,alpha=1)
        if self.mazeAgent.mazeType == 'loop':
            x = self.mazeAgent.discreteCoords[10,:,0]
        if fitEllipse_ == True: 
            (X,Y,Z),_,_= fitEllipse(rf,coords=self.mazeAgent.discreteCoords)
            ax.contour(X, Y, Z, levels=[1], colors=('w'), linewidths=2, linestyles="dashed")
        if show==False:
            plt.close(fig)
        if save == True:
            saveFigure(fig, "receptiveField")
        return fig, ax
    

    def plotGridField(self, hist_id=-1, time=None, fig=None, ax=None, number=0, show=True, animationCall=False, plotTimeStamp=False,save=True,STDP=False):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
        if STDP==True: 
            M = snapshot['W'].T    
        t = snapshot['t'] / 60
        extent = snapshot['mazeState']['extent']
        if hist_id == -1 and animationCall == False:
            gridFields = self.mazeAgent.gridFields
        else:
            gridFields = self.mazeAgent.getGridFields(M=M,alignToFinal=True)

        def sigmoid(x):
            return np.exp(x) / (np.exp(x) + np.exp(-x))
        if number == 'many': 
            fig = plt.figure(figsize=(10, 10*((extent[3]-extent[2])/(extent[1]-extent[0]))))
            gs = matplotlib.gridspec.GridSpec(6, 6, hspace=0.1, wspace=0.1)
            c=0
            # numberstoplot = np.array([60 + 5*i for i in np.arange(36)])
            numberstoplot = np.concatenate((np.array([0,1,2,3,4,5]),np.geomspace(6,gridFields.shape[0]-1,30).astype(int)))
            for i in range(6):
                for j in range(6):
                    ax = plt.subplot(gs[i,j])
                    # ax.imshow(sigmoid(gridFields[numberstoplot[c]]),extent=extent,interpolation=None)
                    ax.imshow(gridFields[numberstoplot[c]],extent=extent,interpolation=None)
                    ax.grid(False)
                    ax.axis('off')
                    ax.text(extent[1]-0.07, extent[3]-0.05,str(numberstoplot[c]+1),fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
                    c+=1

        else:
            #if a figure/ax objects are passed, clear the axis and replot the maze
            if (ax is not None) and (fig is not None): 
                ax.clear()
                self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
            # else if they are not passed plot the maze
            if (fig, ax) == (None, None):
                fig, ax = self.plotMazeStructure(hist_id=hist_id)
            
            if number == None: number = np.random.randint(a=0,b=self.mazeAgent.stateSize-1)

            ax.imshow(gridFields[number],extent=extent,interpolation=None)

            if plotTimeStamp == True: 
                ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
            if show==False:
                plt.close(fig)
        
        if save==True:
            saveFigure(fig, "gridField")
        return fig, ax
        
    def plotFeatureCells(self, hist_id=-1,textlabel=True,shufflebeforeplot=False,centresOnly=False,onepink=False,threetypes=False, save =False):
        fig, ax = self.plotMazeStructure(hist_id=hist_id)
        centres = self.mazeAgent.centres.copy()
        ids = np.arange(len(centres))
        if shufflebeforeplot==True:
            np.random.shuffle(ids)
        centres = centres[ids]
        for (i, centre) in enumerate(centres):
            # if i%10==0:
                if textlabel==True:
                    ax.text(centre[0],centre[1],str(ids[i]),fontsize=8,horizontalalignment='center',verticalalignment='center')
                if self.mazeAgent.mazeType == 'TMaze':
                    if abs(centre[1]-1)<0.001: color = 'C0'
                    elif centre[1]>1.001: color = 'C1'
                    elif centre[1]<0.999: color = 'C2'
                    else: color = 'C3'
                else:
                    color = 'C'+str(i)
                if centresOnly == True: 
                    alpha=1
                    c='darkgrey'
                    if onepink == True:
                        if i == 30:
                            if self.mazeAgent.mazeType == 'twoRooms':
                                centre = np.array([3,1.1])
                                c = 'C3'
                    if self.mazeAgent.mazeType == 'twoRooms': 
                        s = 100; linewidth = 2
                    else: 
                        s = 8; linewidth = 0.8 # used to be 200 and 4 in Tom's code
                    if threetypes==True: 
                        alpha=0.7
                        if i%3 == 0:
                            centre -= [0.02,-0.03]
                            c='C0'
                        elif i%3 == 1:
                            c='C1'
                        elif i%3 == 2:
                            centre += [0.02,-0.03]
                            c='C3'

                    ax.scatter(centre[0],centre[1],marker='x',s=s,color=c,linewidth=linewidth,alpha=alpha)
                else:
                    circle = matplotlib.patches.Ellipse((centre[0],centre[1]), 2*self.mazeAgent.sigmas[i], 2*self.mazeAgent.sigmas[i], alpha=0.5, facecolor=color)
                    ax.add_patch(circle)
        if save == True:        
            saveFigure(fig, "basis")
        return fig, ax 
    
    def plotHeatMap(self,smoothing=1):
        posdata = np.stack(self.mazeAgent.history['pos']) # get position data from history and stack into array
        bins = [int(n/smoothing) for n in list(self.mazeAgent.discreteCoords.shape[:2])] # create bins for histogram based on discrete coordinates and smoothing parameter
        bins.reverse() # reverse bins to match x and y dimensions
        hist = np.histogram2d(posdata[:,0],posdata[:,1],bins=bins)[0] # create 2D histogram of position data with specified bins
        fig, ax = self.plotMazeStructure(hist_id=-1)
        ax.imshow(hist.T, extent=self.mazeAgent.extent) # plot heatmap of histogram transposed to match x and y axes, with extent of maze
        return fig, ax

    def decoderHeatmap(self,
                       condition='theta',
                       bins=40,
                       min_bin_samples=1,
                       vmax_angle=180.0,
                       create_if_missing=False,
                       phase_threshold=1.5 * np.pi,
                       t_window=None,
                       future_window=0.1,
                       sample_every=0.05,
                       min_late_spikes=1,
                       return_data=False):
        """
        Plot spatial decoder error and decoding coverage using cached decoder validation.

        This method reuses the timeseries produced by MazeAgent.validate_late_phase_decoder,
        so it does not rerun spike decoding when cache is available.

        Args:
            condition: 'theta', 'notheta', or 'scrambled'.
            bins: Number of bins per axis, or tuple (nx, ny).
            min_bin_samples: Minimum evaluation samples required to keep a bin.
            vmax_angle: Upper color limit for angle-error panel in degrees.
            create_if_missing: If True, run validation once to populate cache when absent.
            phase_threshold, t_window, future_window, sample_every, min_late_spikes:
                Only used when create_if_missing=True.
            return_data: If True, return computed maps and edges with figure objects.

        Returns:
            fig, ax tuple; optionally with a data dictionary when return_data=True.
        """
        cache = getattr(self.mazeAgent, 'decoder_validation_cache', {})
        cached = cache.get(condition, None)

        if cached is None and create_if_missing:
            res = self.mazeAgent.validate_late_phase_decoder(
                condition=condition,
                phase_threshold=phase_threshold,
                t_window=t_window,
                future_window=future_window,
                sample_every=sample_every,
                min_late_spikes=min_late_spikes,
                return_timeseries=True,
            )
            cache = getattr(self.mazeAgent, 'decoder_validation_cache', {})
            cached = cache.get(condition, None)
            if cached is None and ('timeseries' in res):
                cached = {
                    'timeseries': res['timeseries'],
                    'phase_threshold': float(phase_threshold),
                    't_window': float(2 / self.mazeAgent.thetaFreq if t_window is None else t_window),
                    'future_window': float(future_window),
                    'sample_every': float(sample_every),
                    'min_late_spikes': int(min_late_spikes),
                }

        if cached is None or ('timeseries' not in cached):
            raise ValueError(
                f"No cached decoder validation for condition '{condition}'. "
                "Run validate_late_phase_decoder(..., return_timeseries=True) first "
                "or call decoderHeatmap with create_if_missing=True."
            )

        ts = cached['timeseries']
        if ts is None or len(ts) == 0:
            raise ValueError(f"Cached decoder validation for condition '{condition}' is empty.")

        if hasattr(bins, '__len__'):
            nx, ny = int(bins[0]), int(bins[1])
        else:
            nx = ny = int(bins)

        x_edges = np.linspace(float(self.mazeAgent.extent[0]), float(self.mazeAgent.extent[1]), nx + 1)
        y_edges = np.linspace(float(self.mazeAgent.extent[2]), float(self.mazeAgent.extent[3]), ny + 1)

        if ('pos_x' in ts.columns) and ('pos_y' in ts.columns):
            pos_x = ts['pos_x'].to_numpy(dtype=float)
            pos_y = ts['pos_y'].to_numpy(dtype=float)
        else:
            # Backward compatibility with older cached timeseries.
            hist_times = self.history['t'].to_numpy(dtype=float)
            hist_pos = np.stack(self.history['pos'].to_numpy())
            eval_times = ts['t'].to_numpy(dtype=float)
            right = np.searchsorted(hist_times, eval_times, side='left')
            right = np.clip(right, 0, len(hist_times) - 1)
            left = np.clip(right - 1, 0, len(hist_times) - 1)
            choose_left = np.abs(hist_times[left] - eval_times) <= np.abs(hist_times[right] - eval_times)
            nearest_idx = np.where(choose_left, left, right)
            pos_x = hist_pos[nearest_idx, 0]
            pos_y = hist_pos[nearest_idx, 1]

        decoded = ts['decoded'].to_numpy(dtype=bool)
        angles = ts['angle_deg'].to_numpy(dtype=float)

        valid_pos = np.isfinite(pos_x) & np.isfinite(pos_y)
        eval_count = np.histogram2d(pos_x[valid_pos], pos_y[valid_pos], bins=[x_edges, y_edges])[0]

        valid_dec = valid_pos & decoded & np.isfinite(angles)
        decoded_count = np.histogram2d(pos_x[valid_dec], pos_y[valid_dec], bins=[x_edges, y_edges])[0]
        angle_sum = np.histogram2d(
            pos_x[valid_dec],
            pos_y[valid_dec],
            bins=[x_edges, y_edges],
            weights=angles[valid_dec],
        )[0]

        with np.errstate(invalid='ignore', divide='ignore'):
            angle_error_map = angle_sum / decoded_count
            coverage_map = decoded_count / eval_count

        angle_error_map[decoded_count <= 0] = np.nan
        coverage_map[eval_count <= 0] = np.nan

        # histogram2d returns [x_bin, y_bin]; transpose for image orientation.
        angle_error_map = angle_error_map.T
        coverage_map = coverage_map.T
        eval_count = eval_count.T
        decoded_count = decoded_count.T

        if min_bin_samples > 1:
            low_sample = eval_count < float(min_bin_samples)
            angle_error_map[low_sample] = np.nan
            coverage_map[low_sample] = np.nan

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

        self.plotMazeStructure(fig=fig, ax=ax[0], hist_id=-1)
        im0 = ax[0].imshow(
            angle_error_map,
            origin='lower',
            extent=self.mazeAgent.extent,
            cmap='magma',
            vmin=0,
            vmax=float(vmax_angle),
        )
        ax[0].set_title(f'{condition}: mean angle error (deg)')
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        self.plotMazeStructure(fig=fig, ax=ax[1], hist_id=-1)
        im1 = ax[1].imshow(
            coverage_map,
            origin='lower',
            extent=self.mazeAgent.extent,
            cmap='viridis',
            vmin=0,
            vmax=1,
        )
        ax[1].set_title(f'{condition}: decoding coverage')
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if return_data:
            data = {
                'condition': condition,
                'angle_error_map': angle_error_map,
                'coverage_map': coverage_map,
                'sample_count_map': eval_count,
                'decoded_count_map': decoded_count,
                'x_edges': x_edges,
                'y_edges': y_edges,
                'statistic': 'mean',
            }
            return fig, ax, data

        return fig, ax



    def animateField(self, number=0,field='place',interval=100):
        if field == 'place':
            fig, ax = self.plotPlaceField(hist_id=0,number=number,show=False,save=False)
            anim = FuncAnimation(fig, self.plotPlaceField, fargs=(None, fig, ax, number, False, True, True, False), frames=len(self.snapshots), repeat=False, interval=interval)
        elif field == 'grid':
            fig, ax = self.plotGridField(hist_id=0,number=number,show=False,save=False)
            anim = FuncAnimation(fig, self.plotGridField, fargs=(None, fig, ax, number, False, True, True, False), frames=len(self.snapshots), repeat=False, interval=interval)
        elif field == 'M':
            fig, ax = self.plotM(hist_id=0,show=False,save=False,colorbar=False)
            anim = FuncAnimation(fig, self.plotM, fargs=(None, fig, ax, False,"", "Synaptic Weight Matrix \n (STDP Hebbian learning)", False,False,False,True), frames=len(self.snapshots), repeat=False, interval=interval)
        
        today = datetime.strftime(datetime.now(),'%y%m%d')
        now = datetime.strftime(datetime.now(),'%H%M')
        saveFigure(anim,saveTitle=field+"Animation",anim=True)
        return anim
    
    def animateTrajectory(self, startTime=0, endTime=2, fps=10, sampleRate=10, speedUp= 5, save= False, agentSize=0.15, trailAlpha=0.7, showGoal=True):
        """
        Creates an animation of the trajectory of the agent in the maze.
        Args:
            startTime (float): Start time of the trajectory in minutes.
            endTime (float): End time of the trajectory in minutes.
            fps (int): Frames per second for the animation.
            sampleRate (int): keep every Nth frame to reduce memory usage.
            speedUp (int): Factor to speed up the animation -> by default 5x faster than actual trajectory.
            save (bool): Whether to save the animation as a file.
        """
        # get trajectory data from history
        hist = self.mazeAgent.history

        # `runDecisionTask()` returns times in seconds, but some notebook cells pass minutes.
        # Prefer seconds when the values fall inside the recorded time range, otherwise treat them as minutes.
        hist_t = hist['t'].to_numpy(dtype=float)
        hist_t_max = float(np.max(hist_t)) if len(hist_t) else 0.0
        start_sec = float(startTime) if float(startTime) <= hist_t_max else float(startTime) * 60.0
        end_sec = float(endTime) if float(endTime) <= hist_t_max else float(endTime) * 60.0

        startid = hist['t'].sub(start_sec).abs().to_numpy().argmin()
        endid = hist['t'].sub(end_sec).abs().to_numpy().argmin()
        if endid <= startid:
            raise ValueError(
                f"animateTrajectory got an empty time window: start={startTime}, end={endTime}. "
                "Pass seconds from runDecisionTask() or a valid later end time."
            )

        trajectory = np.stack(hist['pos'][startid:endid])[::sampleRate] # get positions between startid and endid, then skip some to avoid overcrowding the plot

        # get decoded directions for time window, sampled as the trajectory
        decoded_dirs_raw = self.mazeAgent.decoded_dir_history[startid:endid]
        if len(decoded_dirs_raw) == 0:
            # no decoded directions available; create empty 2D array
            decoded_dirs = np.zeros((0, 2))
        else:
            decoded_dirs = np.array(decoded_dirs_raw)[::sampleRate]
            # ensure decoded_dirs is 2D (shape: (n_samples, 2))
            if decoded_dirs.ndim == 1:
                decoded_dirs = decoded_dirs.reshape(-1, 2)

        # build a motion-direction fallback so arrows still appear when the decoder is silent
        motion_dirs = np.zeros_like(trajectory)
        if len(trajectory) > 1:
            motion_dirs[:-1] = trajectory[1:] - trajectory[:-1]
            motion_dirs[-1] = motion_dirs[-2]
        motion_norms = np.linalg.norm(motion_dirs, axis=1, keepdims=True)
        motion_norms[motion_norms < 1e-12] = 1e-12
        motion_dirs = motion_dirs / motion_norms

        # filter out NaNs in decoded dirs (create mask that tracks which trajectory positions have valid decoded directions)
        valid_mask = np.zeros(len(trajectory), dtype=bool)
        if len(decoded_dirs) > 0:
            valid_count = min(len(decoded_dirs), len(valid_mask))
            valid_mask[:valid_count] = ~np.any(np.isnan(decoded_dirs[:valid_count]), axis=1)

        # set up figure and axis
        fig, ax = self.plotMazeStructure(hist_id=-1) # plot maze structure for background
        ax.set_aspect('equal') # this does: equal scaling for x and y axes, so circles look like circles, not ellipses

        if showGoal == True:
            goal = self.plotGoal(fig=fig, ax=ax) # plot goal location

        # creat little agent wedge/arrow patch (starts invisible bcse it will be moved in animation)
        agent_wedge = patches.Wedge((0, 0), agentSize, 0, 60, ec='red', fc='red', alpha=0.8)
        ax.add_patch(agent_wedge)

        # create a scatter trail (starts empty) so frames show an accumulating dotted path
        trail_positions = []
        trail = ax.scatter([], [], s=10, alpha=trailAlpha, color='C0', zorder=2)

        # Only open a video writer when the caller explicitly wants an MP4.
        if save:
            try:
                import shutil
                ffmpeg_exe = shutil.which('ffmpeg')
                if not ffmpeg_exe:
                    for candidate in ('/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg'):
                        if os.path.exists(candidate):
                            ffmpeg_exe = candidate
                            break
                if ffmpeg_exe:
                    os.environ.setdefault('IMAGEIO_FFMPEG_EXE', ffmpeg_exe)
            except Exception:
                pass
        writer = imageio.get_writer('trajectory.mp4', fps=fps) if save else None
        current_quiver = None

        for i, pos in enumerate(trajectory):
            if current_quiver is not None:
                current_quiver.remove()

            # draw quiver arrow for decoded direction if available, otherwise show actual motion direction
            if valid_mask[i]:
                decoded_dir = decoded_dirs[i]
                arrow_dir = decoded_dir
            else:
                arrow_dir = motion_dirs[i]

            current_quiver = ax.quiver(pos[0], pos[1], arrow_dir[0], arrow_dir[1], angles='xy', scale_units='xy', scale=1, color='C1', width=0.005, zorder=3)
            
            # update wedge position to current position in trajectory and orient it along motion
            agent_wedge.set_center(pos)

            # determine heading from next or previous sample (robust at ends)
            if i < len(trajectory) - 1: 
                direction = trajectory[i + 1] - pos # trajectory[i + 1] is the next position, so direction points from current to next
            elif i > 0:
                direction = pos - trajectory[i - 1] # trajectory[i - 1] is the previous position, so direction points from previous to current
            else:
                direction = np.array([1.0, 0.0])
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0])

            heading_angle = np.degrees(np.arctan2(direction[1], direction[0])) + 180 # add 180 to flip wedge to point in direction of motion
            agent_wedge.set_theta1(heading_angle - 30)
            agent_wedge.set_theta2(heading_angle + 30)

            # append current position to trail and update scatter offsets (dotted, accumulated)
            trail_positions.append(pos)
            trail.set_offsets(np.asarray(trail_positions))

            # render frame to canvas and write to video
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            if writer is not None:
                writer.append_data(image)

        if writer is not None:
            writer.close()
            print("Animation saved to trajectory.mp4")
        plt.close(fig)


    def animate_trajectory_with_agent(self, 
                                   sample_rate=10,  # keep every 10th frame
                                   fps=30,
                                   startTime=0,
                                   endTime=2, 
                                   output_file='trajectory.mp4',
                                   agent_size=0.15,
                                   trail_alpha=0.7):
        """
        Animate agent trajectory with sampled frames and orientation wedge.

        Args:
            agent: MazeAgent object with history
            sample_rate: keep every Nth timestep (reduces memory)
            fps: frames per second for output video
            output_file: path to save MP4
            agent_size: radius of wedge/circle for agent body
            trail_alpha: transparency of trajectory line
        """
    
        # Sample full trajectory
        hist = self.mazeAgent.history
        sampled_indices = np.arange(0, len(hist), sample_rate)
        sampled_pos = np.array([hist.iloc[i]['pos'] for i in sampled_indices])
    
        # Set up figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.mazeAgent.extent[0], self.mazeAgent.extent[1])
        ax.set_ylim(self.mazeAgent.extent[2], self.mazeAgent.extent[3])
        ax.set_aspect('equal')
        ax.invert_yaxis()
    
        # Plot full trajectory path once
        ax.plot(sampled_pos[:, 0], sampled_pos[:, 1], 'b-', alpha=trail_alpha, linewidth=1.5, label='Trajectory')
    
        # Plot maze walls
        for wall in self.mazeAgent.mazeState['walls'].values():
            for segment in wall:
                ax.plot([segment[0, 0], segment[1, 0]], 
                    [segment[0, 1], segment[1, 1]], 'k-', linewidth=2)
    
    # Create wedge/arrow patch for agent (starts invisible)
        agent_wedge = patches.Wedge((0, 0), agent_size, 0, 60, 
                                 ec='red', fc='red', alpha=0.8)
        ax.add_patch(agent_wedge)
    
        # Current position marker
        pos_dot, = ax.plot([], [], 'ro', markersize=8, label='Agent')
        ax.legend()
    
        # Render frames and write to video
        try:
            import shutil
            ffmpeg_exe = shutil.which('ffmpeg')
            if not ffmpeg_exe:
                for candidate in ('/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg'):
                    if os.path.exists(candidate):
                        ffmpeg_exe = candidate
                        break
            if ffmpeg_exe:
                os.environ.setdefault('IMAGEIO_FFMPEG_EXE', ffmpeg_exe)
        except Exception:
            pass
        writer = imageio.get_writer(output_file, fps=fps)
    
        for idx in sampled_indices:
            # Get position and direction at this timestep
            pos = np.array(hist.iloc[idx]['pos'])
        
            # Get direction (approximate from velocity or use stored dir if available)
            if idx > 0:
                prev_pos = np.array(hist.iloc[idx-sample_rate]['pos'])
                direction = pos - prev_pos
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.array([1, 0])
            else:
                direction = np.array([1, 0])
        
            # Calculate heading angle (in degrees)
            heading_angle = np.degrees(np.arctan2(direction[1], direction[0]))
        
            # Update wedge position and orientation
            agent_wedge.set_center(pos)
            agent_wedge.set_theta1(heading_angle - 30)  # ±30° wedge
            agent_wedge.set_theta2(heading_angle + 30)
        
            # Update position dot
            pos_dot.set_data([pos[0]], [pos[1]])
        
            # Render frame to canvas
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        
            writer.append_data(image)
    
        writer.close()
        plt.close()
        print(f"Animation saved to {output_file}")    


    def plotMAveraged(self,time=None, x_ticks=None, plot_no_theta=True, color='C1',ylim=None, renorm=True, return_data=False):
        # only works/defined for open loop maze
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()
            snapshot = self.snapshots.iloc[hist_id]
        else:
            snapshot = self.snapshots.iloc[-1]

        M = snapshot['M'].copy()
        W = snapshot['W'].copy() 
        W_notheta = snapshot['W_notheta'].copy()
        W_scrambled = snapshot['W_scrambled'].copy() # added by Livi
        roll = int(self.mazeAgent.nCells/2)
        M_copy, W_copy, W_notheta_copy, W_scrambled_copy = M.copy(), W.copy(), W_notheta.copy(), W_scrambled.copy()
        for i in range(self.mazeAgent.nCells):
            M_copy[i,:] = np.roll(M[i,:],-i+roll)
            W_copy[i,:] = np.roll(W[i,:],-i+roll)
            W_notheta_copy[i,:] = np.roll(W_notheta[i,:],-i+roll)
            W_scrambled_copy[i,:] = np.roll(W_scrambled[i,:],-i+roll)

        M_av,M_std = np.mean(M_copy,axis=0),np.std(M_copy,axis=0)
        W_av,W_std = np.mean(W_copy,axis=0),np.std(W_copy,axis=0)
        W_notheta_av,W_notheta_std = np.mean(W_notheta_copy,axis=0),np.std(W_notheta_copy,axis=0)
        W_scrambled_av,W_scrambled_std = np.mean(W_scrambled_copy,axis=0),np.std(W_scrambled_copy,axis=0)

        print(f"skew W: {getSkewness(W_av)} vs skew W_notheta: {getSkewness(W_notheta_av)} vs skew M: {getSkewness(M_av)} vs skew W_scrambled: {getSkewness(W_scrambled_av)}")
        # print(f"mass ratio = {np.sum(W_av[:int(len(W_av/2))])/np.sum(W_av[int(len(W_av/2)):])} vs {np.sum(W_notheta_av[:int(len(W_notheta_av/2))])/np.sum(W_notheta_av[int(len(W_notheta_av/2)):])} (no theta)")
        print(f"mass ratio = {np.sum(W_av[:25])/np.sum(W_av[25:])} vs {np.sum(W_notheta_av[:25])/np.sum(W_notheta_av[25:])} (no theta)")

        if renorm == True: 
            W_norm = np.maximum(np.max(W_notheta_av),np.max(W_av))
            M_av,M_std = M_av/(np.max(M_av)), M_std/(np.max(M_av))
            W_av,W_std = W_av/W_norm, W_std/W_norm
            W_notheta_av,W_notheta_std = W_notheta_av/W_norm, W_notheta_std/W_norm
            W_scrambled_av,W_scrambled_std = W_scrambled_av/W_norm, W_scrambled_std/W_norm
        x = self.mazeAgent.centres[:,0]
        x = x-x[roll]

        for i in range(len(x)):
            if x[i] > self.mazeAgent.extent[1]/2:
                x[i] = x[i] - self.mazeAgent.extent[1]

        roll = int(self.mazeAgent.nCells/2)


        fig, ax = plt.subplots(2,1,figsize=(2,2))


        Rs_wav = Rsquared(W,M)
        Rsq_wnothetaav = Rsquared(W_notheta,M)
        Rsq_wscrambledav = Rsquared(W_scrambled,M)

        ax[1].plot(x,M_av,c='C0',linewidth=2, label = r" ")
        ax[0].plot(x,W_av,c=color,label=r" ",linewidth=2)
        # ax[1].plot(x,M_theta_av,c='C0',linewidth=1.5,alpha=0.5,linestyle='dotted')

        ax[1].fill_between(x,M_av+M_std,M_av-M_std,facecolor='C0',alpha=0.2)
        ax[0].fill_between(x,W_av+W_std,W_av-W_std,facecolor=color,alpha=0.2)
        if plot_no_theta == True: 
            ax[0].fill_between(x,W_notheta_av+W_notheta_std,W_notheta_av-W_notheta_std,facecolor=color,alpha=0.2)
            ax[0].plot(x,W_notheta_av,c=color,label=r" ",linewidth=1.5,alpha=0.7,linestyle='--', dashes=(1, 1))
        if self.mazeAgent.W_scrambled is not None: # added by Livi
            ax[0].fill_between(x,W_scrambled_av+W_scrambled_std,W_scrambled_av-W_scrambled_std,facecolor='C3',alpha=0.2)
            ax[0].plot(x,W_scrambled_av,c='C3',label=r" ",linewidth=1.5,alpha=0.7)

        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[0].set_xlim(min(x),max(x))
        ax[1].set_xlim(min(x),max(x))
        if ylim is not None: 
            ax[0].set_ylim(0,ylim)
        else:
            ax[0].set_ylim(min(W_av-W_std),max(W_av+W_std))

        ticks = (x_ticks or [-2,-1,0,1,2])
        ticklabels = [""]*len(ticks)
        ax[0].set_xticks(ticks)
        ax[0].set_xticklabels(ticklabels)
        ax[1].set_xticks(ticks)
        ax[1].set_xticklabels(ticklabels)
        # ax[0].tick_params(width=2,color='darkgrey')
        # ax[1].tick_params(width=2,color='darkgrey')
        plt.grid(False)

        ax[0].spines['left'].set_position('zero')
        # ax[0].spines['left'].set_color('darkgrey')
        ax[0].spines['left'].set_linewidth(2)
        ax[0].spines['right'].set_color('none')        
        ax[0].spines['bottom'].set_position('zero')
        # ax[0].spines['bottom'].set_color('darkgrey')
        ax[0].spines['bottom'].set_linewidth(2)
        ax[0].spines['top'].set_color('none')

        ax[1].spines['left'].set_position('zero')
        # ax[1].spines['left'].set_color('darkgrey')
        ax[1].spines['left'].set_linewidth(2)
        ax[1].spines['right'].set_color('none')        
        ax[1].spines['bottom'].set_position('zero')
        # ax[1].spines['bottom'].set_color('darkgrey')
        ax[1].spines['bottom'].set_linewidth(2)
        ax[1].spines['top'].set_color('none')

        ax[0].legend(frameon=False)    
        ax[1].legend(frameon=False)  

        data_dict = {'x': x.copy(), 
                     'M_av': M_av.copy(), 
                     'M_std': M_std.copy(), 
                     'W_av': W_av.copy(), 
                     'W_std': W_std.copy(), 
                     'W_notheta_av': W_notheta_av.copy(), 
                     'W_notheta_std': W_notheta_std.copy(),
                     'W_scrambled_av': W_scrambled_av.copy(), 
                     'W_scrambled_std': W_scrambled_std.copy()
                     
        } 
     
        if return_data:
            print("returning data dict")
            return data_dict
        
        else: 
            return fig, ax

    def plotMetrics(self,total_time=None, x_ticks=None): # Livi note: not adjusted for scrambled theta
        t         = []

        W_snr         = [] 
        W_notheta_snr = []
        W_scrambled_snr = []

        W_r2 = []
        W_notheta_r2  = []
        W_scrambled_r2  = []

        x = self.mazeAgent.centres[:,0]

        M = rowAlignMatrix(self.mazeAgent.snapshots.iloc[-1]['M'])

        for i in range(len(self.mazeAgent.snapshots)-1):
            snapshot = self.mazeAgent.snapshots.iloc[i]
            time = snapshot['t']
            if time >= 31:
                
                R2_W, R2_Wnotheta, R2_Wscrambled, SNR_W, SNR_Wnotheta, SNR_Wscrambled, skew_W, skew_Wnotheta, skew_M, skew_Wscrambled, peak_W, peak_Wnotheta, peak_M, peak_Wscrambled = self.mazeAgent.getMetrics(time=time)
        
                t.append(time/60)

                W_snr.append(SNR_W)
                W_notheta_snr.append(SNR_Wnotheta)
                W_scrambled_snr.append(SNR_Wscrambled)
                W_r2.append(R2_W)
                W_notheta_r2.append(R2_Wnotheta)
                W_scrambled_r2.append(R2_Wscrambled)

        
        snapshot = self.mazeAgent.snapshots.iloc[-1]
        time = snapshot['t']

        fig, ax = plt.subplots(2,1,figsize=(1,2),sharex=True)

        W_r2, W_notheta_r2, W_scrambled_r2 = np.array(W_r2), np.array(W_notheta_r2), np.array(W_scrambled_r2) # added scrambled by Livi
        thresh = 0.5 
        t_50 = t[np.argmin(np.abs(W_r2 - thresh))] 
        t_50_notheta = t[np.argmin(np.abs(W_notheta_r2 - thresh))] 
        t_50_scrambled = t[np.argmin(np.abs(W_scrambled_r2 - thresh))] #added by Livi
        print(f"t0.5 {t_50:.3f} vs t0.5notheta {t_50_notheta:.3f} vs t0.5scrambled {t_50_scrambled:.3f} (no theta vs scrambled)") # added scrambled by Livi

        end = -1
        if total_time is not None: 
            t_last = np.argmin(np.abs(np.array(t) - total_time))
            end = t_last
        ax[1].plot(t[:end],W_snr[:end],c='C1',linewidth=2, label=r"$\theta$")
        ax[1].plot(t[:end],W_notheta_snr[:end],c='C1',linewidth=1.5,linestyle='--', dashes=(1, 1),alpha=0.7,label=r"No $\theta$")
        ax[1].plot(t[:end],W_scrambled_snr[:end],c='C2',linewidth=1.5,linestyle='-.', dashes=(1, 1),alpha=0.7,label=r"Scrambled $\theta$") #added by Livi

        ax[0].plot(t[:end],W_r2[:end],c='C1',linewidth=2)
        ax[0].plot(t[:end],W_notheta_r2[:end],c='C1',linewidth=1.5,linestyle='--', dashes=(1, 1),alpha=0.7)
        ax[0].plot(t[:end],W_scrambled_r2[:end],c='C2',linewidth=1.5,linestyle='-.', dashes=(1, 1),alpha=0.7) #added by Livi



        ax[1].set_ylim(bottom=0,top=max(W_snr)+0.15)
        ax[0].set_ylim(bottom=0,top=1)

        ticks = (x_ticks or [0,15,30])
        ticklabels = [""]*len(ticks)
        ax[1].set_xticks(ticks)
        ax[1].set_xticklabels(ticklabels)
        ax[0].set_xticks(ticks)
        ax[0].set_xticklabels(ticklabels)

        # ax[1].tick_params(width=2,color='darkgrey')
        # ax[0].tick_params(width=2,color='darkgrey')
        ax[1].set_yticks([0,3,6])
        ax[1].set_yticklabels(["","",""])
        ax[0].set_yticks([0,0.5,1])
        ax[0].set_yticklabels(["","",""])

        for i in range(2):

            ax[i].spines['left'].set_position('zero')
            # ax[i].spines['left'].set_color('darkgrey')
            ax[i].spines['left'].set_linewidth(2)
            ax[i].spines['right'].set_color('none')        
            ax[i].spines['bottom'].set_position('zero')
            # ax[i].spines['bottom'].set_color('darkgrey')
            ax[i].spines['bottom'].set_linewidth(2)
            ax[i].spines['top'].set_color('none')
        
        return fig, ax 


    def plotFieldSilhouette(self, N=25, plot_pf=True, plot_pf_notheta=False, plot_pf_M=True, plot_rf=False, plot_pf_scrambled=False, no_theta=False, ax=None):  
        # take the snapshot at 30 minutes (or closest to it)  
        hist_id = self.snapshots['t'].sub(30*60).abs().to_numpy().argmin()
        snapshot = self.snapshots.iloc[hist_id-22]

        # get the x coordinates of the maze, the receptive field and place fields for the Nth cell, and normalize them by their integral (trapezoidal rule) to get comparable curves
        x = self.mazeAgent.discreteCoords[10,:,0]
        rf = self.mazeAgent.discreteStates[10,:,N]
        pf = self.mazeAgent.getPlaceFields(M=self.mazeAgent.W, threshold=0)[N][10,:]
        pf_notheta = self.mazeAgent.getPlaceFields(M=snapshot['W_notheta'], threshold=0)[N][10,:]
        pf_scrambled = self.mazeAgent.getPlaceFields(M=snapshot['W_scrambled'], threshold=0)[N][10,:] # added by Livi   
        pf_M = self.mazeAgent.getPlaceFields(M=self.mazeAgent.M, threshold=0)[N][10,:]
        rf, pf, pf_notheta, pf_scrambled, pf_M = rf/np.trapezoid(rf,x), pf/np.trapezoid(pf,x), pf_notheta/np.trapezoid(pf_notheta,x), pf_scrambled/np.trapezoid(pf_scrambled,x), pf_M/np.trapezoid(pf_M,x) #added scrambled here by Livi

        if ax is None:
            fig, ax = plt.subplots(figsize=(2,0.5))
        else:
            ax.clear()
            fig = ax.figure

        ax.set_xlim(0,5)
        if plot_rf == True:
            ax.fill_between(x[rf>=0],rf[rf>=0],0,facecolor="#02a3a6",alpha=0.5)
        if plot_pf_M == True:
            ax.fill_between(x[pf_M>=0],pf_M[pf_M>=0],0,facecolor="#8c52ff",alpha=0.5)        
        if plot_pf == True:
            ax.fill_between(x[pf>=0],pf[pf>=0],0,facecolor="#ff66c4",alpha=0.5)
        if plot_pf_notheta == True:
            ax.fill_between(x[pf_notheta>=0],pf_notheta[pf_notheta>=0],0,facecolor="#cb6ce6",alpha=0.5)
        if plot_pf_scrambled == True:
            ax.fill_between(x[pf_scrambled>=0],pf_scrambled[pf_scrambled>=0],0,facecolor="#e2a9f1",alpha=0.5) #added by Livi

        pf = self.mazeAgent.getPlaceFields(M=self.mazeAgent.W, threshold=0)[N]
        pf_notheta = self.mazeAgent.getPlaceFields(M=snapshot['W_notheta'], threshold=0)[N]
        pf_M = self.mazeAgent.getPlaceFields(M=self.mazeAgent.M, threshold=0)[N]
        pf_scrambled = self.mazeAgent.getPlaceFields(M=snapshot['W_scrambled'], threshold=0)[N] #added by Livi
        print("R2: M-->W=",Rsquared(pf,pf_M),"M-->Wnotheta=",Rsquared(pf_notheta,pf_M),"M-->Wscrambled=",Rsquared(pf_scrambled,pf_M)) # added scrambled by Livi
        
        if hasattr(tpl, 'xyAxes'): 
            tpl.xyAxes(ax)
        else:
            # fall back option if xyAxes in tpl is again not available (added by Livi)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax.spines['left'].set_color('none')
        ax.set_xticks([0,2.5,5])
        ax.set_yticks([])
        ax.set_xticklabels(["","",""])
        if ax is None:
            plt.tight_layout()

        return fig, ax
    
    def plotGoal(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = self.plotMazeStructure()
        goal = self.mazeAgent.goalPos
        r = self.mazeAgent.goalRadius
        ax.scatter(goal[0], goal[1], c="#ff5ebb", s= 8, marker="*", label="Reward") # original parameters: s=800, c="#ff5ebb", marker="*", label="Reward"
        ax.add_patch(matplotlib.patches.Circle((goal[0], goal[1]), r, fill=False, edgecolor="#ff5ebb")) # original parameters: linewidth=2
        ax.legend(frameon=False, labelcolor='#ff5ebb') # original parameters: fontsize=40, labelcolor='#ff5ebb'
        return fig, ax
    
    def valueFunctionHeatMap(self, fig=None, ax=None, smoothing=1, showGoal=True, showCentres = False, save=False):
        """
        Plot a smooth value field over the maze grid using precomputed cell values.
        Args:
            smoothing (float, optional): Gaussian sigma in grid bins for display smoothing.
            showGoal (bool, optional): when True, goal position also plotted in heatmap
            showCentres (bool, optional): when True, cell centres also plotted in heatmap
        """
        cellValues = np.asarray(self.mazeAgent.cellValues, dtype=float).reshape(-1)
        # print max value in cellValues
        #print(np.max(cellValues), np.min(cellValues))
        centres = np.asarray(self.mazeAgent.centres, dtype=float)
        if cellValues.size != centres.shape[0]:
            raise ValueError("cellValues length must match number of cell centres.")

        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=-1)

        # Normalize the basis states at each grid location to sum to 1, then project cell values through these normalized states to get a smooth value field.
        states = self.mazeAgent.discreteStates
        denom = np.sum(states, axis=2, keepdims=True)
        denom[denom == 0] = 1.0
        states_norm = states / denom
        value_field = np.einsum("ijk,k->ij", states_norm, cellValues)
        value_field = np.nan_to_num(value_field, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
        value_field = value_field.astype(float)
        # Project cell values through the basis states to obtain a smooth value at each grid location.
        #value_field = np.einsum("ijk,k->ij", self.mazeAgent.discreteStates, cellValues)
        #print("value_field min,max:", np.nanmin(value_field), np.nanmax(value_field))

        sigma = float(smoothing) if smoothing is not None else 0.0
        if sigma > 0:
            from scipy.ndimage import gaussian_filter
            value_field = gaussian_filter(value_field, sigma=sigma)

        # mask values outside of the maze walls by creating a mask
        # grid over the image
        x = np.linspace(self.mazeAgent.extent[0], self.mazeAgent.extent[1], value_field.shape[1])
        y = np.linspace(self.mazeAgent.extent[2], self.mazeAgent.extent[3], value_field.shape[0])
        X, Y = np.meshgrid(x, y)

        # TMaze geometry
        rs = self.mazeAgent.roomSize
        hw = 0.05 * rs
        arm_w = 2 * hw

        inside = (
            ((X >= 0) & (X <= rs) & (Y >= rs - hw) & (Y <= rs + hw)) |
            ((X >= rs) & (X <= rs + arm_w) & (Y >= 0) & (Y <= 2 * rs))
        )

        # if you smooth first, re-apply the mask afterwards
        masked_field = np.ma.array(value_field, mask=~inside)

        cmap = plt.cm.magma.copy()
        cmap.set_bad(alpha=0)

        im = ax.imshow(
            masked_field,
            extent=self.mazeAgent.extent,
            cmap=cmap,
            alpha=1.0,
            interpolation='bilinear'
        )  
        #print("im.get_clim():", im.get_clim()) 
        
        #im = ax.imshow(
            #value_field,
            #extent=self.mazeAgent.extent,
            #cmap='cool',
            #alpha=0.8,
            #interpolation='bilinear'
        #)
    
        if showCentres == True:
            ax.scatter(centres[:,0], centres[:,1], c='white', marker='x') # original parameters: s=25
        if showGoal == True:
            self.plotGoal(fig, ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, label='Value', orientation='horizontal') # aspect
        cbar.ax.tick_params() # original parameters: labelsize=50
        cbar.ax.set_xticks([0, 0.9])  
        cbar.ax.set_xticklabels(["min", "max"]) # original parameters: fontsize=50
        cbar.ax.set_ylim(bottom=0, top=0.9)
        cbar.set_label('Value', fontweight='bold') # original parameters: fontsize=60, fontweight='bold'
        if save == True:
            saveFigure(fig, "valueFunctionHeatmap")
        return fig, ax

## Helper functions

def fitEllipse(image, threshold=0.5,coords=None,verbose=True):
    """Takes an array (image) and fits an ellipse to it. 
    It does this by finding contours then regressing these points against the formula Ax2 + Bxy + Cy2 + Dx + Ey = 1

    Args:
        image (np.array()): The image or array to which the 
        threshold (float, optional): The relativethreshold upon whch the edges of the image will be defined. Defaults to 0.75.
        coords (np.array(image.shape,2)): The underlying coordinates of the image . Defaults to None in which case tries to get them .

    Returns:
        tuple of arrays: the x, y and z coords of the ellipse function (can be contour plotted on top of image)
    """ 

    fig2, ax2 = plt.subplots()
    cs = ax2.contour(coords[...,0],coords[...,1],image,np.array([threshold*max(image.flatten())])).collections[0].get_paths()[0].vertices
    plt.close()
    x,y = cs[:,0], cs[:,1]
    X = np.stack((x**2,x*y,y**2,x,y)).T
    F = 1
    coords_ = coords.reshape(-1,2)
    xmin,xmax,ymin,ymax=min(coords_[:,0]),max(coords_[:,0]),min(coords_[:,1]),max(coords_[:,1])
    Y = F*np.ones(X.shape[0])
    (A,B,C,D,E) = np.matmul(np.linalg.inv(np.matmul(X.T,X) + 0.0*np.identity(X.T.shape[0])),np.matmul(X.T,Y)) #least squares fit ellipse Ax2 + Bxy + Cy2 + Dx + Ey = F
    x_coord = np.linspace(xmin,xmax,1000)
    y_coord = np.linspace(ymin,ymax,1000)   
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = A * X_coord ** 2 + B * X_coord * Y_coord + C * Y_coord**2 + D * X_coord + E * Y_coord 
    #finally get eccentricity 
    m = np.array([[A,   B/2,  D/2],
                    [B/2, C,    E/2],
                    [D/2, E/2,  F]])
    if np.linalg.det(m) < 0:
        eta = 1
    else:
        eta = -1
    eccen = np.sqrt((2*np.sqrt((A-C)**2 + B**2))/(eta*(A+C)+np.sqrt((A-C)**2 + B**2)))
    if verbose == True:
        print("Eccentricity = %.3f" %eccen)

    angle = np.arctan((1/B)*(C-A-np.sqrt((A-C)**2+B**2)))

    return (X_coord, Y_coord, Z_coord), eccen, angle

def rowAlignMatrix(M):
    M_copy = M.copy()
    roll = int(M.shape[0]/2)
    for i in range(M.shape[0]):
        M_copy[i,:] = np.roll(M[i,:],-i+roll)
    return M_copy

def saveFigure(fig,saveTitle="",transparent=True,anim=False,specialLocation=None,figureDirectory="../figures/"):
    """saves figure to file, by data (folder) and time (name) 
    Args:
        fig (matplotlib fig object): the figure to be saved
        saveTitle (str, optional): name to be saved as. Current time will be appended to this Defaults to "".
    """	

    today =  datetime.strftime(datetime.now(),'%y%m%d')
    if not os.path.isdir(figureDirectory + f"{today}/"):
        os.mkdir(figureDirectory + f"{today}/")
    figdir = figureDirectory + f"{today}/"
    now = datetime.strftime(datetime.now(),'%H%M')
    path_ = f"{figdir}{saveTitle}_{now}"
    path = path_
    i=1
    while True:
        if os.path.isfile(path+".pdf") or os.path.isfile(path+".mp4"):
            path = path_+"_"+str(i)
            i+=1
        else: break
    if anim == True:
        fig.save(path + ".mp4")
    else:
        fig.savefig(path+".pdf", dpi=400,transparent=transparent)
    
    if specialLocation is not None: 
        fig.savefig(specialLocation, dpi=400,transparent=transparent,bbox_inches='tight')

    return path

def pickleAndSave(class_,name,saveDir='../savedObjects/'):
	"""pickles and saves a class
	this is not an efficient way to save the data, but it is easy 
	this will overwrite previous saves without warning
	Args:
		class_ (any class): the class/model to save
		name (str): the name to save it under 
		saveDir (str, optional): Directory to save into. Defaults to './savedItems/'.
	"""	
	with open(saveDir + name+'.pkl', 'wb') as output:
		dill.dump(class_, output)
	return 

def loadAndDepickle(name, saveDir='../savedObjects/'):
	"""Loads and depickles a class saved using pickleAndSave
	Args:
		name (str): name it was saved as
		saveDir (str, optional): Directory it was saved in. Defaults to './savedItems/'.
	Returns:
		class: the class/model
	"""	
	with open(saveDir + name+'.pkl', 'rb') as input:
		item = dill.load(input)
	return item

def Rsquared(y1, y2):
    """R squared between two arrays 

    Args:
        y1 (np.array()): array 1
        y2 (np.array()): array 2

    Returns:
        float: R squared value between them 
    """    
    return ((1/y1.size) * np.sum((y1-np.mean(y1)) * (y2-np.mean(y2))) / (np.std(y1) * np.std(y2)))**2


def getCOM(array):
    print(array.shape)
    i_av, j_av = 0, 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            i_av += array[i,j]*i
            j_av += array[i,j]*j
    i_av /= np.sum(array)
    j_av /= np.sum(array)
    i_av = int(i_av)
    j_av = int(j_av)
    return (i_av,j_av)

def getMoment(x,y,moment=1,c=0):
    """Get moments not from sample of points but from a function (list of x, and f(x)=y)

    Args:
        x (np.array): independent variable
        y (np.array): dependent variable
        moment (int, optional): which moment ot get. Defaults to 1.
        c (float): about which point to find moment, defaults to zero
    """    
    s_x = 0
    s_y = 0
    for i in range(len(x)):
        s_x += ( (x[i] - c)**moment ) * y[i]
        s_y += y[i]
    return s_x / s_y

def getCircularMoment(theta,y,moment=1,c=0):
    Cp = np.sum(np.cos(moment*theta)*y) / np.sum(y)
    Sp = np.sum(np.sin(moment*theta)*y) / np.sum(y)
    Rp = np.sqrt(Cp**2 + Sp**2)
    if Cp > 0 and Sp > 0: 
        Tp = np.arctan(Sp/Cp)
    elif Cp < 0: 
        Tp = np.arctan(Sp/Cp) + np.pi
    elif Sp < 0 and Cp > 0: 
        Tp = np.arctan(Sp/Cp) + 2*np.pi
    return Rp, Tp



def getSkewness(y,circular=False):
    if not np.all(y>=0):
        y = np.maximum(y,0)
    x = np.linspace(0,2*np.pi,len(y))
    if circular == False: 
        mean = getMoment(x,y)
        std = np.sqrt(getMoment(x,y,moment=2,c=mean))
        skewness = getMoment(x,y,moment=3,c=mean) / std**3
    if circular == True: #NCSS Statistical Software NCSS.com, Chapter 230, Circular Data Analysis, https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Circular_Data_Analysis.pdf
        R1, T1 = getCircularMoment(x,y,moment=1)
        R2, T2 = getCircularMoment(x,y,moment=2)
        V = 1-R1
        skewness = R2*np.sin(2*T1-T2) / (1-R1)**(3/2)

    return skewness


def getPeak(x,y,smooth=True):
    if smooth == True: 
        y_smooth = np.empty_like(y)
        for i in range(len(y)):
            y_smooth[i] = np.mean(y[max(0,i-1):min(i+2,len(y))])
        peak = x[np.argmax(y_smooth)]
    else:
        peak = x[np.argmax(y)]
    return peak 

def fisherZ(r):
    return np.log((1+r)/(1-r)) / 2


def ornstein_uhlenbeck(dt, x, drift=0.0, noise_scale=0.2, coherence_time=5.0):
    """An ornstein uhlenbeck process in x.
    x can be multidimensional 
    Args:
        dt: update time step
        x: the stochastic variable being updated
        drift (float, or same type as x, optional): [description]. Defaults to 0.
        noise_scale (float, or same type as x, optional): Magnitude of deviations from drift. Defaults to 0.2 (20 cm s^-1 if units of x are in metres).
        coherence_time (float, optional): Effectively over what time scale you expect x to change directions. Defaults to 5.
    Returns:
        dx (same type as x); the required update to x
    """
    x = np.array(x)
    drift = drift * np.ones_like(x)
    noise_scale = noise_scale * np.ones_like(x)
    coherence_time = coherence_time * np.ones_like(x)
    sigma = np.sqrt((2 * noise_scale ** 2) / (coherence_time * dt))
    theta = 1 / coherence_time
    dx = theta * (drift - x) * dt + sigma * np.random.normal(size=x.shape, scale=dt)
    return dx