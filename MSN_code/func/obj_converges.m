%% obj_converges: function description
function [converged] = obj_converges(obj_values, absTol)
converged = false;
if length(obj_values) < 3
    converged = false;
else
    current = obj_values(length(obj_values));
    last = obj_values(length(obj_values) - 1);
    if (last - current) / last < absTol
        converged = true;
    end
end
end