--[[
   Training the model parameters
   
   @author: Sagar M. Waghmare
--]]

require 'math'
require 'xlua'
require 'optim'

-- Cuda
local useCuda = opt.useCuda
local deviceId = opt.deviceId

if useCuda then
   -- Check if cunn is installed
   local hasCuda = pcall(function() require 'cunn' end)
   if not hasCuda then
      error('Cuda/cunn not installed.')
      return
   end
end

-- Optimization and Testing functions
local optimizer = require 'optimizer'

local noValidation = opt.noValidation

-- Confusion matrix
local classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
local confusion = optim.ConfusionMatrix(classes)

-- if true use best train/validation model for testing
local best = opt.best

   -- Learning
local batchSize = opt.batchSize
local epochs = opt.epochs
local maxTries = opt.maxTries
local learningRate = opt.learningRate
local learningRateDecay = opt.learningRateDecay
local linearDecay = opt.linearDecay
local startEpoch = opt.startEpoch
local endLearningRate = opt.endLearningRate
assert(epochs > startEpoch, "startEpoch should be smaller than epochs.")

local momentum = opt.momentum
local loss = opt.loss
local adam = opt.adam
local onlyPrintAttemptAccu = opt.onlyPrintAttemptAccu

-- Linearly reduce the learning rate
local learningRates = torch.Tensor()
if linearDecay then
   if verbose then print("Using linear decay.") end
   learningRates:resize(startEpoch):fill(learningRate)
   local temp = torch.range(learningRate, endLearningRate,
                            -learningRate/(epochs-startEpoch))
   learningRates = torch.cat(learningRates, temp)
end

-- If true use Adaptive moment estimation else SGD.
if adam then
   if verbose then print("Using Adaptive moment estimation optimizer.") end
   optimMethod = optim.adam
else
   if verbose then print("Using Stocastic gradient descent optimizer.") end
   optimMethod = optim.sgd
end
if verbose then
   print(optimMethod)
end

local attempts = opt.attempts
local testAccus = torch.zeros(attempts) -- testing accuracy
local validAccus = torch.zeros(attempts) -- validation accuracy
for attempt=1,attempts do

   -- For each attempt we change the seed of pseudo random number generator
   torch.manualSeed(attempt)

   -- Model and Criterions
   -- get randomly initialized model for each attempt
   model, criterions = dofile('model.lua')

   if verbose then
      print(model)
      print(criterions)
   end

   -- Learning hyperparameters
   -- Keeping inside for loop optimState's value changes after every epoch
   local optimState = {
                       coefL1 = 0,
                       coefL2 = 0,
                       learningRate = learningRate,
                       weightDecay = 0.0,
                       momentum = momentum,
                       learningRateDecay = learningRateDecay
                      }

   if useCuda then
      if verbose then print("Using GPU: "..deviceId) end
      cutorch.setDevice(deviceId)
      if verbose then print("GPU set") end

      model:cuda()
      if verbose then print("Model copied to GPU.") end

      criterions:cuda()
      if verbose then print("Criterion copied to GPU.") end
   else
      if verbose then print("Not using GPU.") end
   end

   -- Retrieve parameters and gradients
   parameters, gradParameters = model:getParameters()

   -- Additional variables for keeping track of training
   local displayProgress = verbose
   local classifierIndx = 1
   local trainAccu = 0
   local validAccu = 0
   local bestTrainAccu = 0
   local bestValidAccu = 0
   local trainLoss = 0
   local validLoss = 0
   local bestTrainLoss = math.huge
   local bestValidLoss = math.huge
   local bestTrainModel = nil
   local bestValidModel = nil
   local earlyStopCount = 0

   for i=1, epochs do
      if linearDecay then
         optimState.learningRate = learningRates[i]
      end
      -- Training
      trainLoss = optimizer:model_train_multi_criterion(model, criterions,
                                              parameters, gradParameters,
                                              trData, optimMethod, optimState,
                                              batchSize, i, confusion,
                                              trainLogger, useCuda,
                                              displayProgress, classiferIndx)
      confusion:updateValids()
      if loss then
         if verbose then
            print("Current train loss: ".. trainLoss
                     ..", best train loss: " .. bestTrainLoss)
         end
         if trainLoss < bestTrainLoss then
            bestTrainLoss = trainLoss
            bestTrainModel = model:clone()
            if verbose then print(confusion) end
            validAccus[attempts] = confusion.totalValid * 100
         end
      else -- Using classification accuracy for saving best train model
         trainAccu = confusion.totalValid * 100
         if bestTrainAccu < trainAccu then
            bestTrainAccu = trainAccu
            bestTrainModel = model:clone()
            bestTrainLoss = trainLoss
            validAccus[attempt] = trainAccu
         end
         if verbose then
            print("Current train accu: ".. trainAccu
                     ..", best train accu: " .. bestTrainAccu
                     ..", best train loss: " .. bestTrainLoss)
         end
      end

      -- Validating
      if not noValidation then
         validLoss = optimizer:model_test_multi_criterion(model, criterions,
                                                tvData, confusion,
                                                useCuda, classifierIndx)
         confusion:updateValids()
         if loss then
            if verbose then
               print("Current valid loss: ".. validLoss
                        ..", best valid loss: " .. bestValidLoss)
            end
            if validLoss < bestValidLoss then
               earlyStopCount = 0
               bestValidLoss = validLoss
               bestValidModel = model:clone()
               if verbose then print(confusion) end
               validAccus[attempts] = confusion.totalValid * 100
            else
               earlyStopCount = earlyStopCount + 1
            end
         else
            validAccu = confusion.totalValid * 100
            if bestValidAccu < validAccu then
               earlyStopCount = 0
               bestValidAccu = validAccu
               bestValidModel = model:clone()
               bestValidLoss = validLoss
               validAccus[attempt] = validAccu
            else
               earlyStopCount = earlyStopCount + 1
            end
            if verbose then
               print("Current valid accu: ".. validAccu
                     ..", best valid accu: " .. bestValidAccu
                     ..", best valid loss: " .. bestValidLoss)
            end
         end
         if verbose then
            print(noiseSigma, weightTied, useBatchNorm, eta, earlyStopCount)
         end
      end

      if maxTries ~= 0 then
         if earlyStopCount >= maxTries then
            if verbose then print("Early stopping at epoch: " .. i) end
            break
         end
      end
   end

   -- Testing
   if best then
      if noValidation then
         testLoss = optimizer:model_test_multi_criterion(bestTrainModel,
                                               criterions, tsData, confusion,
                                               useCuda, classifierIndx)
      else
         testLoss = optimizer:model_test_multi_criterion(bestValidModel,
                                               criterions, tsData, confusion,
                                               useCuda, classifierIndx)
      end
   else
      testLoss = optimizer:model_test_multi_criterion(model, criterions,
                                            tsData, confusion,
                                            useCuda, classifierIndx)
   end
   confusion:updateValids()
   testAccu = confusion.totalValid * 100
   testAccus[attempt] = testAccu
   if verbose or onlyPrintAttemptAccu then
      print("Attempt: " .. tostring(attempt) .. " Test Accu: " .. testAccu)
   end
end
return testAccus, validAccus
