#include <iostream>

// dx = * d(fk)/d(dh)
class DHCalibrator{

    float error(){

    }

    float step(){
        // y = fk(theta), A = angles
            // psi = Jacobian(y, dh)
        float err = error();
        // d_dh = inv(dot(psi.T, psi)) * psi.T * err
        // dh += d_dh
    }

};
