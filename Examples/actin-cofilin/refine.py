# import section
command = xplor.command
from jCoupPot import JCoupPot
from noePot import NOEPot
import prePot
from xplor import select
from xplorPot import XplorPot
from rdcPotTools import *
from pdbTool import *
from atomAction import *
from selectTools import *
from simulationTools import *
from ivm import IVM
import protocol
import monteCarlo
protocol.initRandomSeed()   #set random seed - by time

# step1 structure define
protocol.initParams('./parallhdg_new.pro')
protocol.initStruct('input/actin-prepare.psf',erase=False)
protocol.initCoords('input/actin-prepare.pdb')
AtomSel("all").apply( SetProperty('segmentName', 'ALT1') )
protocol.initStruct('input/cofilin-prepare.psf',erase=False)
protocol.initCoords('input/cofilin-prepare.pdb')
AtomSel("all and not (segid ALT1)").apply( SetProperty('segmentName', 'BLT1') )
protocol.initStruct('input/cofilin-prepare.psf',erase=False)
protocol.initCoords('input/cofilin-prepare.pdb')
AtomSel("all and not (segid ALT1 or segid BLT1)").apply( SetProperty('segmentName', 'BLT2') )
protocol.initNBond(repel=1.2)

# step2 interaction define
command("""

    constraints

    inter = (segid ALT1)(segid BLT1)
    inter = (segid ALT1)(segid BLT2)
    weights * 1 end end

    """)

if xplor.p_processID==0:
  command("write psf output=complex.psf end")

# step3 annealing settings

init_t  = 3000
final_t = 25

cool_steps = 12000

from simulationTools import MultRamp, StaticRamp, InitialParams
rampedParams=[]

potList = PotList()
potList.add( XplorPot("BOND") )

potList.add( XplorPot("ANGL") )
rampedParams.append( MultRamp(0.4,1,"potList['ANGL'].setScale(VALUE)") )

potList.add( XplorPot("IMPR") )
rampedParams.append( MultRamp(0.4,1,"potList['IMPR'].setScale(VALUE)") )

potList.add( XplorPot("VDW") )
rampedParams.append( MultRamp(1.2,0.75,
                              "command('param nbonds repel VALUE end end')") )
rampedParams.append( MultRamp(.004,4,
                              "command('param nbonds rcon VALUE end end')") )

# step4 restraints define
noe=PotList('noe')
potList.append(noe)
from noePotTools import create_NOEPot
pot = create_NOEPot('xlms',"./input/xlms.tbl")
pot.setPotType("hard")
pot.setScale(2)       
pot.setAveType("sum")
noe.append(pot)
rampedParams.append( MultRamp(2,30, "noe.setScale( VALUE )") )

# step5 IVM setup

dyn  = IVM()

dyn.fix("""segid ALT1 """)
dyn.group(""" segid BLT1 """)
dyn.group(""" segid BLT2 """)

# step6 sampling and output

def structLoopAction(loopInfo):

    protocol.initMinimize(dyn, potList=potList)
    InitialParams( rampedParams )
    dyn.run()

    # high temp dynamics

    ini_timestep = 0.010
    potList["VDW"].setScale(0)
    protocol.initDynamics(dyn,
                          potList=potList,
                          bathTemp=init_t,
                          initVelocities=True,
                          stepsize=ini_timestep,
                          finalTime=10,
                          printInterval=100)
    dyn.run()

    # cooling

    timestep=ini_timestep
    potList["VDW"].setScale(1)

    protocol.initDynamics(dyn,
                          potList=potList,
                          bathTemp=init_t,
                          initVelocities=True,
                          stepsize=timestep,
                          finalTime=0.5,
                          printInterval=100)

    dyn.setResetCMInterval( 100 )

    AnnealIVM(initTemp =init_t,
              finalTemp=final_t,
              numSteps = 50,
              ivm=dyn,
              rampedParams = rampedParams).run()
    # final Powell minimization
    protocol.initMinimize(dyn)
    dyn.run()

    loopInfo.writeStructure(potList)

    pass

StructureLoop(numStructures=480,
              pdbTemplate='./output/Calc_STRUCTURE.pdb',
              structLoopAction=structLoopAction).run()
