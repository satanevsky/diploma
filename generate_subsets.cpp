#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <algorithm>
using namespace std;

vector<vector<int> > v;
int threshold;


void input() {
    ifstream in("input.txt");
    int n, m;
    in >> n >> m >> threshold;
    //threshold = 1;
    v = vector<vector<int> >(n, vector<int>(m, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            in >> v[i][j];
        }
    }
}

vector<vector<int> > get_start() {
    vector<vector<int> > start;
    cout << v.size() << ' ' << v[0].size() << endl;
    for (int i = 0; i < v[0].size(); ++i) {
        int cnt1 = 0, cnt0 = 0;
        for (int j = 0; j < v.size(); ++j) {
            if (v[j][i] == 1) ++cnt1;
	    if (v[j][i] == 0) ++cnt0;
        }
        
        if (cnt1 >= threshold) {
            vector<int> cand(2, -1);
            cand[1] = i;
            start.push_back(cand);
        }
    }
    return start;
}


void gen() {
    ofstream fout("output.txt");
    vector<vector<int> > start = get_start();
    int ans_cnt = start.size();
    cout << start.size() << endl;
    while(start.size() > 0) {
        cout << "new iteration " << start.size();
        cout.flush();
        map<vector<int>, vector<int> > d;
        for (int i = 0; i < start.size(); ++i) {
            auto start_cut = start[i];
            start_cut.pop_back();
            d[start_cut].push_back(start[i][start[i].size() - 1]);
        }

        vector<vector<int> > new_start;
        map<vector<int>, vector<vector<int> > > val_ind;
	int n_val = 0;
        for (map<vector<int>, vector<int> >::iterator it = d.begin(); it != d.end(); ++it) {
            for (int i = 0; i < it->second.size(); ++i) {
                for (int j = i + 1; j < it->second.size(); ++j) {
                    vector<int> cand = it->first;
                    cand.push_back(it->second[i]);
                    cand.push_back(it->second[j]);
                    vector<int> cur_ind;
		    for (int i = 0; i < v.size(); ++i) {
			bool good = true;
			for (int j = 1; j < cand.size(); ++j) {
				if (v[i][cand[j]] != 1) {
				   good = false;
				   break;
				}
			}
			if (good) {
				cur_ind.push_back(i);
			}
		     }
	             if (cur_ind.size() >= threshold) {
			val_ind[cur_ind].push_back(cand);
			n_val++;
		     }
		}
	    }
        }
        cout << ' ' << n_val;
        cout.flush();
        start.clear();
	
	cout << " prunning ";
	cout.flush();
	for (auto it = val_ind.begin(); it != val_ind.end(); ++it) {
	    if (it->second.size() == 1) {
		start.push_back(it->second[0]);
            } else {
		vector<int> ans;
		for (int i = 0; i < it->second.size(); ++i) {
			vector<int> temp_ans(ans.size() + it->second[i].size());
			merge(ans.begin(), ans.end(), it->second[i].begin(), it->second[i].end(), temp_ans.begin());
			ans.clear();
                        for (int j = 0; j < temp_ans.size(); ++j) {
				if (j == 0 || temp_ans[j-1] != temp_ans[j]) ans.push_back(temp_ans[j]);
			}
		}
		start.push_back(ans);
	    }
	    fout << it->first.size() << ' ';
	    for (int i = 0; i < it->first.size(); ++i) {
		fout << it->first[i] << ' ';
            }
            fout << '\n';
	}
        ans_cnt += start.size();
        cout << ' ' << start.size() << ' ' << ans_cnt << endl;
    }
}


int main() {
    cout << "hello" << endl;
    input();
    cout << "input ok" << endl;
    gen();
    return 0;
}
