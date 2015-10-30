function z = ShlU(z,th)
z = sign(z).*max(0,abs(z)-th);
end

