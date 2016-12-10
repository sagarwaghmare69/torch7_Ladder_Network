--[[
   Hudl Assignment. 
   Implementatiing ladder network for supervised classification with
   denoising as auxillary task.
   Ref: http://arxiv.org/pdf/1504.08215.pdf

   Sample run command:
   th main.lua --verbose --eta 500 --epochs 100 --learningRate 0.002 --linearDecay --endLearningRate 0 --startEpoch 50 --useCuda --deviceId 1 --noiseSigma 0.3 --useBatchNorm --batchSize 100 --adam --noValidation --attempts 10 --useCuda
   
   @author: Sagar M. Waghmare
--]]

require 'math'

torch.setdefaulttensortype("torch.FloatTensor")

cmd = torch.CmdLine()
cmd:text()
cmd:text()
local titleMsg = 'Hudl assignment: Supervised learning with'..
                  'unsupervised learning as auxillary cost'
cmd:text(titleMsg)
cmd:text()
cmd:text()

-- Data
-- Default using 20% of training data for validation
cmd:option('--noValidation', false,
           'Use validation data for training as well.')
cmd:option('--best', false,
           'Use best training or validation model for tesing.')

-- Model parameters
cmd:option('--noOfClasses', 10, 'Number of classes.') -- MNIST data
cmd:option('--noiseSigma', 0,
           'Stdev for noise for denoising autoencoder (Mean is zero).')
cmd:option('--hiddens', '{1000, 500, 250, 250, 250}', 'Hiddens units')
cmd:option('--useBatchNorm', false, 'Use batch normalization')
cmd:option('--weightTied', false, 'Tie weights of encoder and decoder')

-- Criterion and learning
cmd:option('--attempts', 1, 'Run [attempts] independent experiments.')
cmd:option('--eta', 0, 'If zero then only classifier cost is considered.')
cmd:option('--batchSize', 32, 'Batch Size.')
cmd:option('--epochs', 100, 'Number of epochs.')
cmd:option('--maxTries', 0, 'Number of tries for stopping.')
cmd:option('--learningRate', 0.002, 'Learning rate')
cmd:option('--learningRateDecay', 1e-07, 'Learning rate decay')
cmd:option('--linearDecay', false, 'Linearly reduce learning rate')
cmd:option('--startEpoch', 1, 'Epoch number when to start linear decay.')
cmd:option('--endLearningRate', 0, 'Learning rate at last epoch')
cmd:option('--momentum', 0, 'Learning Momemtum')
cmd:option('--loss', false, 
           'If true use loss for early stopping else confusion matrix.')
cmd:option('--adam', false, 'Use adaptive moment estimation optimizer.')

-- Use Cuda
cmd:option('--useCuda', false, 'Use GPU')
cmd:option('--deviceId', 1, 'GPU device Id')

-- Print debug messages
cmd:option('--verbose', false, 'Print apppropriate debug messages.')
cmd:option('--onlyPrintAttemptAccu', false,
           'Only print test accuracy for every attempt.')

-- Command line arguments
opt = cmd:parse(arg)
print(opt)

-- Fixing seed to faciliate reproduction of results.
torch.manualSeed(-1)

verbose = opt.verbose

--MNIST datasource
ds, trData, tvData, tsData = dofile('datasource.lua')
if verbose then
   print(trData)
   print(tvData)
   print(tsData)
end

-- Training
testAccus, validAccus = dofile('train.lua')
print("Test accuracies.")
print(testAccus)
print("Max Test Error is: " .. tostring(100 - testAccus:max()) .. "%")
print("Avg Test Error is: " .. tostring(100 - testAccus:mean()) .. "%")
