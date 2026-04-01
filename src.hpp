// Heuristic digit recognizer for MNIST-like 28x28 grayscale images.
// Implements judge(IMAGE_T&) without any I/O.
#ifndef SRC_HPP_HEURISTIC_JUDGE_030
#define SRC_HPP_HEURISTIC_JUDGE_030

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <cstdint>

typedef std::vector<std::vector<double> > IMAGE_T;

namespace nr_heuristic {

struct BBox { int r0, c0, r1, c1; int h() const { return r1 - r0 + 1; } int w() const { return c1 - c0 + 1; } };

static inline double clamp01(double x){ if(x<0) return 0; if(x>1) return 1; return x; }

static double otsu_threshold(const IMAGE_T &img){
    // 256-bin Otsu on [0,1]
    const int bins = 256;
    std::vector<double> hist(bins, 0.0);
    int H = (int)img.size(); if(H==0) return 0.5; int W = (int)img[0].size(); if(W==0) return 0.5;
    double total = (double)H * W;
    for(int r=0;r<H;++r){
        const auto &row = img[r];
        for(int c=0;c<W;++c){
            int b = (int)std::floor(clamp01(row[c]) * (bins-1));
            hist[b] += 1.0;
        }
    }
    for(int i=0;i<bins;++i) hist[i] /= total;
    std::vector<double> P(bins,0.0), M(bins,0.0);
    double mG = 0.0;
    for(int i=0;i<bins;++i){
        P[i] = hist[i] + (i?P[i-1]:0.0);
        M[i] = i*hist[i] + (i?M[i-1]:0.0);
        mG += i*hist[i];
    }
    double best_sigma = -1.0; int best_t = bins/2;
    for(int t=0;t<bins-1;++t){
        double w0 = P[t];
        double w1 = 1.0 - w0;
        if(w0<=1e-9 || w1<=1e-9) continue;
        double m0 = M[t] / w0;
        double m1 = (mG - M[t]) / w1;
        double sigma_b = w0 * w1 * (m0 - m1)*(m0 - m1);
        if(sigma_b > best_sigma){ best_sigma = sigma_b; best_t = t; }
    }
    return (double)best_t / (bins-1);
}

static BBox find_bbox(const std::vector<std::vector<uint8_t> > &bin){
    int H = (int)bin.size(); int W = H? (int)bin[0].size():0;
    int r0=H, c0=W, r1=-1, c1=-1;
    for(int r=0;r<H;++r){
        for(int c=0;c<W;++c){
            if(bin[r][c]){
                if(r<r0) r0=r; if(r>r1) r1=r; if(c<c0) c0=c; if(c>c1) c1=c;
            }
        }
    }
    if(r1<r0 || c1<c0){ return {0,0,H-1,W-1}; }
    return {r0,c0,r1,c1};
}

static int count_holes(const std::vector<std::vector<uint8_t> > &bin, const BBox &b){
    // Count background components inside bbox that do NOT touch bbox border
    int H = (int)bin.size(); int W = H? (int)bin[0].size():0;
    if(H==0||W==0) return 0;
    int h=b.h(), w=b.w();
    std::vector<uint8_t> vis((size_t)h*w, 0);
    auto idx=[&](int r,int c){ return (r-b.r0)*w + (c-b.c0); };
    auto inb=[&](int r,int c){ return r>=b.r0 && r<=b.r1 && c>=b.c0 && c<=b.c1; };
    int holes=0;
    const int dr[8]={-1,-1,-1,0,0,1,1,1};
    const int dc[8]={-1,0,1,-1,1,-1,0,1};
    for(int r=b.r0;r<=b.r1;++r){
        for(int c=b.c0;c<=b.c1;++c){
            if(bin[r][c]) continue; // background only
            int id = idx(r,c);
            if(vis[id]) continue;
            std::queue<std::pair<int,int>> q;
            q.emplace(r,c); vis[id]=1;
            bool touches_border=false;
            while(!q.empty()){
                auto rc = q.front(); q.pop();
                int rr = rc.first, cc = rc.second;
                if(rr==b.r0 || rr==b.r1 || cc==b.c0 || cc==b.c1) touches_border=true;
                for(int k=0;k<8;++k){
                    int nr=rr+dr[k], nc=cc+dc[k];
                    if(!inb(nr,nc)) continue;
                    if(bin[nr][nc]) continue; // only background
                    int nid = idx(nr,nc);
                    if(!vis[nid]){ vis[nid]=1; q.emplace(nr,nc);}    
                }
            }
            if(!touches_border) ++holes;
        }
    }
    return holes;
}

struct Features{
    double density;
    double aspect;
    int holes;
    double cx, cy;
    double top_ratio, bottom_ratio, left_ratio, right_ratio;
    double q[4];
    double top_band_ratio;
    double mid_hband_ratio;
};

static Features compute_features(const std::vector<std::vector<uint8_t> > &bin){
    int H=(int)bin.size(); int W=H? (int)bin[0].size():0;
    Features f{}; if(H==0||W==0){ return f; }
    BBox b = find_bbox(bin);
    int h=b.h(), w=b.w();
    double total=0.0; double sumr=0.0, sumc=0.0;
    for(int r=b.r0;r<=b.r1;++r){
        for(int c=b.c0;c<=b.c1;++c){
            if(bin[r][c]){ total+=1.0; sumr += (r-b.r0); sumc += (c-b.c0); }
        }
    }
    if(total<=0.0){ f.density=0.0; f.aspect= (double)w/std::max(1,h); f.holes=0; f.cx=0.5; f.cy=0.5; return f; }
    f.density = total / (w*h);
    f.aspect = (double)w / (double)h;
    f.holes = count_holes(bin, b);
    f.cx = sumc / total / std::max(1,w-1);
    f.cy = sumr / total / std::max(1,h-1);

    double top=0,bottom=0,left=0,right=0;
    int r_mid = b.r0 + h/2;
    int c_mid = b.c0 + w/2;
    double qsum[4]={0,0,0,0};
    for(int r=b.r0;r<=b.r1;++r){
        for(int c=b.c0;c<=b.c1;++c){
            if(!bin[r][c]) continue;
            if(r<r_mid) top++; else bottom++;
            if(c<c_mid) left++; else right++;
            int qi = (r<r_mid?0:2) + (c<c_mid?0:1);
            qsum[qi] += 1.0;
        }
    }
    f.top_ratio = top/total; f.bottom_ratio=bottom/total; f.left_ratio=left/total; f.right_ratio=right/total;
    for(int i=0;i<4;++i) f.q[i] = qsum[i]/total;

    int top_h = std::max(1, (int)std::floor(0.2*h));
    int mid0 = b.r0 + (int)std::floor(0.4*h);
    int mid1 = b.r0 + (int)std::ceil(0.6*h);
    double topband=0, midband=0;
    for(int r=b.r0;r<b.r0+top_h && r<=b.r1;++r){
        for(int c=b.c0;c<=b.c1;++c) if(bin[r][c]) topband++;
    }
    for(int r=mid0;r<=mid1 && r<=b.r1;++r){
        for(int c=b.c0;c<=b.c1;++c) if(bin[r][c]) midband++;
    }
    f.top_band_ratio = topband/total;
    f.mid_hband_ratio = midband/total;
    return f;
}

static int classify_with_rules(const std::vector<std::vector<uint8_t> > &bin){
    Features f = compute_features(bin);

    if(f.holes >= 2){
        return 8;
    }
    if(f.holes == 1){
        if(f.top_ratio - f.bottom_ratio > 0.15) return 9;
        if(f.bottom_ratio - f.top_ratio > 0.15) return 6;
        if(std::abs(f.aspect - 1.0) < 0.2) return 0;
        if(f.cy < 0.5) return 9; else return 6;
    }

    if(f.aspect < 0.55 && std::abs(f.cx - 0.5) < 0.12 && f.density < 0.6){
        return 1;
    }

    if(f.top_band_ratio > 0.34 && f.bottom_ratio < 0.48 && f.right_ratio > f.left_ratio){
        return 7;
    }

    if(f.mid_hband_ratio > 0.26 && f.q[1] > f.q[0] && f.q[2] < 0.18){
        return 4;
    }

    if( (f.right_ratio - f.left_ratio) > 0.18 && f.q[2] < 0.16 ){
        return 3;
    }
    if( f.q[2] > f.q[3] + 0.06 && f.top_band_ratio > 0.25 ){
        return 5;
    }
    if( f.top_ratio > f.bottom_ratio && (f.q[3] - f.q[2]) > 0.04 ){
        return 2;
    }

    if(f.density > 0.55 && std::abs(f.aspect-1.0) < 0.25) return 0;
    if(f.right_ratio > f.left_ratio + 0.1) return 9;
    if(f.top_ratio > f.bottom_ratio + 0.1) return 7;
    return 2;
}

} // namespace nr_heuristic

int judge(IMAGE_T &img){
    int H = (int)img.size(); if(H==0) return 0; int W = (int)img[0].size(); if(W==0) return 0;
    double t = nr_heuristic::otsu_threshold(img);
    std::vector<std::vector<uint8_t> > bin(H, std::vector<uint8_t>(W, 0));
    for(int r=0;r<H;++r){
        for(int c=0;c<W;++c){
            double v = nr_heuristic::clamp01(img[r][c]);
            bin[r][c] = (uint8_t)(v >= t ? 1 : 0);
        }
    }
    bool any=false; for(int r=0;r<H && !any;++r) for(int c=0;c<W;++c) if(bin[r][c]) { any=true; break; }
    if(!any){
        for(int r=0;r<H;++r) for(int c=0;c<W;++c) bin[r][c] = (uint8_t)(img[r][c] >= 0.5 ? 1 : 0);
    }
    int pred = nr_heuristic::classify_with_rules(bin);
    if(pred < 0 || pred > 9) pred = 0;
    return pred;
}

#endif // SRC_HPP_HEURISTIC_JUDGE_030
