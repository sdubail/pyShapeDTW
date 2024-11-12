% extract features for a time series sequence

function [feats, statactivefeatNames, physactivefeatNames]= ...
                    featureExtractor(seq, ...
                                statfeatureLists, statfield2function, ...
                                  physfeatureLists, physfield2function)

% parameter setting, which features will be extracted

%{
    activeAxis = {'X', 'Y', 'Z'};
    statfeatureLists = struct('mean',      true, ...
                               'std',       true, ...
                               'rms',       true, ...
                               'meanderivative',    true, ...
                               'meancrossingrate',  true);
    statfield2function = struct('mean',             'meanSeq', ...
                            'std',              'stdSeq', ...
                            'rms',              'rmsSeq', ...
                            'meanderivative',   'meanDerivative', ...
                            'meancrossingrate', 'meancrossingRate');


    physfeatureLists = struct('meanMI', true, ...
                          'stdMI', true, ...
                          'NSM', true, ...
                          'accelEnergy', true, ...
                          'corrcoef', true);

    physfield2function = struct('meanMI', 'meanMotionIntensity', ...
                            'stdMI', 'stdMI', ...
                            'accelEnergy', 'accelEnergy', ...
                            'corrcoef', 'corrcoef');

 %} 

    
    [nDims, nAxis] = size(seq);
    if nDims < nAxis
        warning('Make sure the time series is a column vector\n');
    end

    % (2) =========== extracting statistical features ==============
    statfeatures = [];
    for i=1:nAxis
       [statfeats, statactivefeatNames] = statfeatureExtractor(seq(:,i), ...
                            statfeatureLists, statfield2function); 
        statfeatures = [statfeatures; statfeats];
    end
  
    
    % (3) ============ extracting physical features ===============
    [physfeatures, physactivefeatNames] = physfeatureExtractor(seq, ...
                            physfeatureLists, physfield2function);
     
    feats = [statfeatures; physfeatures];
    feats = feats';
end