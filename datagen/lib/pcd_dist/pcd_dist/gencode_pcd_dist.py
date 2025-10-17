from asc_autogen.stub_functions.stub_fun_910b import *

BASEM = 1024


class CustCube(CubeModule):
    def initialize(self):
        ...

    def forward(self):
        cube_ready()
        wait_vec()


class CustVec(VecModule):
    def initialize(self, M, K):
        self.M = M  # number of queries
        self.K = K  # number of targets
        self.MPERCORE_UPPER = CeilDiv(CeilDiv(self.M, BASEM), GetVecNum()) * BASEM
        self.M1 = self.MPERCORE_UPPER * GetVecIdx()
        self.M2 = Min(self.M1 + self.MPERCORE_UPPER, self.M)

        self.xdata = Tensor(DT.float, BASEM)
        self.ydata = Tensor(DT.float, BASEM)
        self.zdata = Tensor(DT.float, BASEM)

        self.xbuf = DBuff(DT.float, BASEM)
        self.ybuf = DBuff(DT.float, BASEM)
        self.zbuf = DBuff(DT.float, BASEM)
        self.prevdistbuf = DBuff(DT.float, BASEM)
        self.distbuf = Tensor(DT.float, BASEM)

        self.xcurr = Tensor(DT.float, 64)
        self.ycurr = Tensor(DT.float, 64)
        self.zcurr = Tensor(DT.float, 64)

        self.distoutbuf = DBuff(DT.float, BASEM)

    def forward(self, x, y, z, x0, y0, z0, distout):
        cnt = self.create_var('cnt', DT.int, 0)
        xanchor = self.create_var('xanchor', DT.float, 0)
        yanchor = self.create_var('yanchor', DT.float, 0)
        zanchor = self.create_var('zanchor', DT.float, 0)

        with Loop('m', self.M1, self.M2, BASEM) as m:
            curr_idx = self.create_var('curr_idx', DT.int, 0)
            with Loop('k', 0, self.K) as k:
                self.wait_in_empty()
                # load queries
                with If(k == 0):
                    with If ((m + BASEM) <= self.M2):
                        gm_to_ub(self.xdata[cnt], x[m], 1, BASEM//8, 0, 0)
                        gm_to_ub(self.ydata[cnt], y[m], 1, BASEM//8, 0, 0)
                        gm_to_ub(self.zdata[cnt], z[m], 1, BASEM//8, 0, 0)
                    with Else():
                        tail_num = self.M2 - m
                        gm_to_ub_pad(self.xdata[cnt], x[m], 1, tail_num * 4, 0, 0, 0, 0, 0, 0)
                        gm_to_ub_pad(self.ydata[cnt], y[m], 1, tail_num * 4, 0, 0, 0, 0, 0, 0)
                        gm_to_ub_pad(self.zdata[cnt], z[m], 1, tail_num * 4, 0, 0, 0, 0, 0, 0)
                # load target
                remain = self.K - (curr_idx // 8) * 8
                with If((remain >= 8)):
                    gm_to_ub(self.xcurr, x0[curr_idx//8*8], 1, 1, 0, 0)
                    gm_to_ub(self.ycurr, y0[curr_idx//8*8], 1, 1, 0, 0)
                    gm_to_ub(self.zcurr, z0[curr_idx//8*8], 1, 1, 0, 0)
                with Else():
                    gm_to_ub_pad(self.xcurr, x0[curr_idx//8*8], 1, remain * 4, 0, 0, 0, 0, 0, 0)
                    gm_to_ub_pad(self.ycurr, y0[curr_idx//8*8], 1, remain * 4, 0, 0, 0, 0, 0, 0)
                    gm_to_ub_pad(self.zcurr, z0[curr_idx//8*8], 1, remain * 4, 0, 0, 0, 0, 0, 0)
                self.set_in_ready()
                # current anchor
                self.wait_in_ready()
                xanchor.GetValue(self.xcurr[curr_idx%8])
                yanchor.GetValue(self.ycurr[curr_idx%8])
                zanchor.GetValue(self.zcurr[curr_idx%8])
                xanchor *= -1
                yanchor *= -1
                zanchor *= -1
                # compute distance
                bar_all()
                adds(self.xbuf[cnt], self.xdata[cnt], xanchor, BASEM//64, 1, 1, 8, 8)
                adds(self.ybuf[cnt], self.ydata[cnt], yanchor, BASEM//64, 1, 1, 8, 8)
                adds(self.zbuf[cnt], self.zdata[cnt], zanchor, BASEM//64, 1, 1, 8, 8)
                bar_v()
                mul(self.xbuf[cnt], self.xbuf[cnt], self.xbuf[cnt], BASEM//64, 1, 1, 1, 8, 8, 8)
                mul(self.ybuf[cnt], self.ybuf[cnt], self.ybuf[cnt], BASEM//64, 1, 1, 1, 8, 8, 8)
                mul(self.zbuf[cnt], self.zbuf[cnt], self.zbuf[cnt], BASEM//64, 1, 1, 1, 8, 8, 8)
                bar_v()
                add(self.distbuf, self.xbuf[cnt], self.ybuf[cnt], BASEM//64, 1, 1, 1, 8, 8, 8)
                bar_v()
                add(self.distoutbuf[cnt], self.distbuf, self.zbuf[cnt], BASEM//64, 1, 1, 1, 8, 8, 8)
                bar_v()
                self.set_in_empty()
                # compare distance
                with If(k != 0):
                    vmin(self.distoutbuf[cnt], self.distoutbuf[cnt], self.prevdistbuf[cnt], BASEM//64, 1, 1, 1, 8, 8, 8)
                    bar_all()
                curr_idx += 1
                ub_to_ub(self.prevdistbuf[cnt], self.distoutbuf[cnt], 1, BASEM//8, 0, 0)
            # write distance
            self.wait_out_empty()
            sqrt(self.distoutbuf[cnt], self.distoutbuf[cnt], BASEM//64, 1, 1, 8, 8)
            bar_all()
            self.set_out_ready()
            self.wait_out_ready()
            with If ((m + BASEM) <= self.M2):
                ub_to_gm(distout[m], self.distoutbuf[cnt], 1, BASEM//8, 0, 0) #TODO: 判断尾块，dist搬出
            with Else():
                tail_num = self.M2 - m
                # pad_num = CeilDiv(tail_num, 8) * 8 - tail_num # max 32Byte, 8对齐
                ub_to_gm_pad(distout[m], self.distoutbuf[cnt], 1, tail_num * 4, 0, 0) # tail
            self.set_out_empty()
            bar_all()
            cnt += 1
            bar_all()


class KernelCust(KernelBase):
    def init(self, M, K):
        self.vec0 = CustVec(M, K)

    def forward(self, x, y, z, x0, y0, z0, distout):
        self.vec0(x, y, z, x0, y0, z0, distout)


if __name__=='__main__':
    M = Var('M', DT.int, value=40960)
    K = Var('K', DT.int, value=1048576)

    xmtx = GMTensor('xmtx', DT.float, length=M)
    ymtx = GMTensor('ymtx', DT.float, length=M)
    zmtx = GMTensor('zmtx', DT.float, length=M)
    x0mtx = GMTensor('x0mtx', DT.float, length=K)
    y0mtx = GMTensor('y0mtx', DT.float, length=K)
    z0mtx = GMTensor('z0mtx', DT.float, length=K)
    outmtx = GMTensor('outmtx', DT.float, length=M, is_output=True)

    kernel = KernelCust(M, K)
    kernel(xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx)

    kernel.gen_code('kernel_custom_dist.cpp', 'kernel_cust_dist')
    kernel.gen_test_entry('main.cpp', kernel_file='kernel_custom_dist.cpp', kernel_name='kernel_cust_dist', n_cores=20, test_speed=True, force=True)
    kernel.gen_golden('golden.py')
    kernel.gen_checker('check.py')

