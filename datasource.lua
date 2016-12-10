--[[
   Datasource for MNIST training, validation and testing data.

   @author: Sagar M. Waghmare
--]]

require 'dp'

local ds = dp.Mnist{}

local noValidation = opt.noValidation
local t1, t2
local trData = {}
local tvData = {}
local tsData = {}

trData.data, t1, t2 = ds:get('train', 'input', 'bf', 'float')
trData.labels, t1, t2 = ds:get('train', 'target')
trData.size = function() return trData.data:size()[1] end

tvData.data, t1, t2 = ds:get('valid', 'input', 'bf', 'float')
tvData.labels, t1, t2 = ds:get('valid', 'target')
tvData.size = function() return tvData.data:size()[1] end

tsData.data, t1, t2 = ds:get('test', 'input', 'bf', 'float')
tsData.labels, t1, t2 = ds:get('test', 'target')
tsData.size = function() return tsData.data:size()[1] end

if noValidation then
   trData.data = torch.cat(trData.data, tvData.data, 1) 
   trData.labels = torch.cat(trData.labels, tvData.labels, 1) 
   tvData.data = nil
   tvData.labels = nil
   tvData.size = nil
end
collectgarbage()

-- returning dataset, trainingData, validationData, testingData
return ds, trData, tvData, tsData
