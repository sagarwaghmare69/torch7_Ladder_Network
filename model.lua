--[[
   Build nngraph module for classifier+denoising_autoencoder
   with lateral connections

   @author: Sagar M. Waghmare
--]]

require 'nn'
require 'dpnn'
require 'nngraph'

-- Model
local noOfClasses = opt.noOfClasses
local noiseSigma = opt.noiseSigma
local inputHiddens =loadstring('return ' .. opt.hiddens)()
local useBatchNorm = opt.useBatchNorm
local weightTied = opt.weightTied

local linFeats = ds:iSize('f')

local hiddens = {linFeats}
for i=1,#inputHiddens do
   hiddens[#hiddens+1] = inputHiddens[i]
end
hiddens[#hiddens+1] = noOfClasses

-- encoder input
local input = nil
if noiseSigma ~= 0 then
   if verbose then print("Adding noise to the samples.") end
   input = nn.WhiteNoise(0, noiseSigma)()
else
   input = nn.Identity()()
end

-- encoder model
local encoderLayers = {}
local Zs = {}
Zs[1] = input
local Hs = {}
Hs[1] = input
for i=2,#hiddens do
   -- Zs
   encoderLayers[i] = nn.Linear(hiddens[i-1], hiddens[i])
   if useBatchNorm then
      Zs[i] = nn.BatchNormalization(hiddens[i])
                                   (encoderLayers[i](Hs[i-1]))
   else
      Zs[i] = encoderLayers[i](Hs[i-1])
   end
  
   -- Hs
   if i==#hiddens then
      Hs[i] = nn.CMul(hiddens[i])(nn.Add(hiddens[i])(Zs[i]))
   else
      Hs[i] = nn.ReLU()(nn.CMul(hiddens[i])(nn.Add(hiddens[i])(Zs[i])))
   end
end

-- classifier
local classifier = nn.LogSoftMax()(Hs[#Hs])

-- Decoder
local decoderLayers = {}
local Z_hats = {}
for i=#hiddens,1,-1 do

   -- u = 0 hence no cij
   if i==#hiddens then
      z_hat1 = nn.CMul(hiddens[i])(Zs[i])
      z_hat2 = nn.CMul(hiddens[i])(Zs[i])
      z_hat3 = nn.CMul(hiddens[i])(Zs[i])
      z_hat34 = nn.Add(hiddens[i])(z_hat3)
      z_hatSigmoid34 = nn.Sigmoid()(z_hat34)
      z_hat234 = nn.CMulTable()({z_hat2, z_hatSigmoid34})
      z_hat5 = nn.CMul(hiddens[i])(Zs[i])
      Z_hats[i] = nn.CAddTable()({z_hat1, z_hat234, z_hat5})
   else
      decoderLayers[i] = nn.Linear(hiddens[i+1], hiddens[i])
      if weightTied then
         if verbose then print("Tying encoder-decoder weights.") end
         decoderLayers[i].weight:set(encoderLayers[i+1].weight:t())
         decoderLayers[i].gradWeight:set(encoderLayers[i+1].gradWeight:t())
      end

      u = decoderLayers[i](Z_hats[i+1])

      cu1 = nn.CMul(hiddens[i])(u)
      du1 = nn.Add(hiddens[i])(u)
      a1 = nn.CAddTable()({cu1, du1})
      cu2 = nn.CMul(hiddens[i])(u)
      du2 = nn.Add(hiddens[i])(u)
      a2 = nn.CAddTable()({cu2, du2})
      cu3 = nn.CMul(hiddens[i])(u)
      du3 = nn.Add(hiddens[i])(u)
      a3 = nn.CAddTable()({cu3, du3})
      cu4 = nn.CMul(hiddens[i])(u)
      du4 = nn.Add(hiddens[i])(u)
      a4 = nn.CAddTable()({cu4, du4})
      cu5 = nn.CMul(hiddens[i])(u)
      du5 = nn.Add(hiddens[i])(u)
      a5 = nn.CAddTable()({cu5, du5})

      z_hat1 = nn.CMulTable()({a1, Zs[i]})
      z_hat2 = nn.CMulTable()({a3, Zs[i]})
      z_hat3 = nn.Sigmoid()(nn.CAddTable()({z_hat2, a4}))
      z_hat4 = nn.CMulTable()({a2, z_hat3})
      Z_hats[i] = nn.CAddTable()({z_hat1, z_hat4, a5})
   end
end
-- Decoder = Z_hats[1]
local model = nn.gModule({input}, {classifier, Z_hats[1]})

-- Criterion and learning
-- Criterion
local eta = opt.eta
local criterions = nn.ParallelCriterion()
local NLL = nn.ClassNLLCriterion() -- Classification criterion for classifier
local MSE = nn.MSECriterion() -- Reconstruction criterion for autoencoder
criterions:add(NLL)
criterions:add(MSE, eta)

return model, criterions
