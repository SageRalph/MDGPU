package mdgpu;
import com.amd.aparapi.Kernel;

public class MDKernel extends Kernel {

private final int N;
private final float[] x;
private final float[] y;
private final float[] ax;
private final float[] ay;
private final float forceCutoff2;

public MDKernel(float[] x, float[] y, float[] ax, float[] ay, float fc, int N) {
    this.x = x;
    this.y = y;
    this.ax = ax;
    this.ay = ay;
    this.forceCutoff2 = fc * fc;
    this.N = N;
}

public void run() {
//    
//    // 2D decomposition 
//    // NOT WORKING - lacks atomic add opperation for accelerations
//
//    final int i = getGlobalId(0);
//    final int j = getGlobalId(1);
//    if (i != j && i != 0) {
//
//        final float dx, dy, dx2, dy2, rSquared, rSquaredInv,
//                attract, repel, fOverR, fx, fy;
//
//        dx = x[i] - x[j];
//        dx2 = dx * dx;
//        if (dx2 < forceCutoff2) { // make sure they're close enough to bother
//            dy = y[i] - y[j];
//            dy2 = dy * dy;
//            if (dy2 < forceCutoff2) {
//                rSquared = dx2 + dy2;
//                if (rSquared < forceCutoff2) {
//                    rSquaredInv = 1.0f / rSquared;
//                    attract = rSquaredInv * rSquaredInv * rSquaredInv;
//                    repel = attract * attract;
//                    fOverR = 24.0f * ((2.0f * repel) - attract) * rSquaredInv;
//                    fx = fOverR * dx;
//                    fy = fOverR * dy;
//    
//                    // NEEDS TO ADD ATOMICALY
//                    ax[i] += fx;  // add this force on to i's acceleration (mass = 1)
//                    ay[i] += fy;
//                }
//            }
//        }
//    }
//
//    

    // 1D decomposition
    //
    float dx, dy, dx2, dy2, rSquared, rSquaredInv,
            attract, repel, fOverR, fx, fy;

    final int i = getGlobalId(0);
    int j;

    for (j = 0; j < N; j++) { // loop over all distinct pairs

        if (i != j) {

            dx = x[i] - x[j];
            dx2 = dx * dx;
            if (dx2 < forceCutoff2) { // make sure they're close enough to bother
                dy = y[i] - y[j];
                dy2 = dy * dy;
                if (dy2 < forceCutoff2) {
                    rSquared = dx2 + dy2;
                    if (rSquared < forceCutoff2) {
                        rSquaredInv = 1.0f / rSquared;
                        attract = rSquaredInv * rSquaredInv * rSquaredInv;
                        repel = attract * attract;
                        fOverR = 24.0f * ((2.0f * repel) - attract) * rSquaredInv;
                        fx = fOverR * dx;
                        fy = fOverR * dy;
                        ax[i] += fx;  // add this force on to i's acceleration (mass = 1)
                        ay[i] += fy;
                    }
                }
            }
        }
    }
}
}
