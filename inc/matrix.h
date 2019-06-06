#pragma once

#include "ftwt.h"

static const uint32_t maxPrint = 10;
template<class T> struct TripletMat;

template<class T>
struct CSCMat
{
    vector<T> vals;
    vector<uint32_t> colIdcs;
    vector<uint32_t> offsets;

    uint32_t n;
    uint32_t m;
    string name;

    CSCMat() : n(0), m(0), name("") {}
    CSCMat(uint32_t rows, uint32_t cols, string name) : n(rows), m(cols), name(name) {}

    void print(bool bAll = false)
    {
        uint32_t r = 0;

        printf("CSC Matrix - %s:\n", name.c_str());

        for (uint32_t i = 0; i < vals.size(); i++)
        {
            if (i >= maxPrint && bAll == false) break;       
            while (i == offsets[r + 1]) r++;
            printf("(%d, %d): %g\n", r, colIdcs[i], vals[i]);
        }

        printf("\n");
    }

    void operator+=(const CSCMat<T>& rhs)
    {
        assert(rhs.n == n && rhs.m == m);

        TripletMat<T> tmat = toTriplet();

        for (uint32_t r = 0; r < rhs.n; r++)
        {
            for (uint32_t i = rhs.offsets[r]; i < rhs.offsets[r + 1]; i++)
            {
                tmat.insert({ r, rhs.colIdcs[i], rhs.vals[i] });
            }
        }

        CSCMat csc  = tmat.toCSC();

        vals        = csc.vals;
        colIdcs     = csc.colIdcs;
        offsets     = csc.offsets;
    }

    vector<T> operator*(const vector<T>& rhs)
    {
        assert(rhs.size() == m);
        vector<T> res(n, 0);

        for (uint32_t r = 0; r < n; r++)
        {
            for (uint32_t i = offsets[r]; i < offsets[r + 1]; i++)
            {
                res[r] += vals[i] * rhs[colIdcs[i]];
            }
        }

        return res;
    }

    TripletMat<T> toTriplet()
    {
        TripletMat<T> tripletMat;
        tripletMat.n = n;
        tripletMat.m = m;

        for (uint32_t r = 0; r < n; r++)
        {
            for (uint32_t i = offsets[r]; i < offsets[r + 1]; i++)
            {
                tripletMat.insert({ r, colIdcs[i], vals[i] });
            }
        }

        return tripletMat;
    }
};

template<class T>
struct Triplet
{
    uint32_t r;
    uint32_t c;
    T val;
};

template<class T>
struct TripletMat
{
    vector<Triplet<T>> entries;
    uint32_t n;
    uint32_t m;
    string name;

    TripletMat() : n(0), m(0), name("") {}
    TripletMat(uint32_t rows, uint32_t cols, string name) : n(rows), m(cols), name(name) {}

    static bool comp(const Triplet<T>& a, const Triplet<T>& b)
    {
        if (a.r < b.r) return true;
        if (a.r == b.r && a.c < b.c) return true;
        return false;
    }

    void insert(Triplet<T> triplet) 
    { 
        if (triplet.r + 1 > n) n = triplet.r + 1;
        if (triplet.c + 1 > m) m = triplet.c + 1;
        entries.push_back(triplet);
    }

    void print(bool bAll = false)
    {
        printf("Triplet Matrix - %s:\n", name.c_str());

        for (uint32_t i = 0; i < entries.size(); i++)
        {
            if (i >= maxPrint && bAll == false) break;
            printf("(%d, %d): %g\n", entries[i].r, entries[i].c, entries[i].val);
        }

        printf("\n");
    }

    void sortAndCombine()
    {
        sort(entries.begin(), entries.end(), comp);
        combineDuplicates();
    }

    CSCMat<T> toCSC()
    {
        CSCMat<T> csc(n, m, name);

        sortAndCombine();

        uint32_t r      = 0;
        uint32_t cnt    = 0;

        csc.offsets.push_back(0);

        for (auto& e : entries)
        {
            while (e.r > r)
            {
                r++;
                csc.offsets.push_back(cnt);
            }

            csc.vals.push_back(e.val);
            csc.colIdcs.push_back(e.c);
            cnt++;
        }

        csc.offsets.push_back((uint32_t)entries.size());

        return csc;
    }

private:

    void combineDuplicates()
    {
        uint32_t cur = 0;

        for (uint32_t i = 1; i < entries.size(); i++)
        {
            if (entries[i].r == entries[cur].r && entries[i].c == entries[cur].c)
            {
                entries[cur].val += entries[i].val;
            }
            else
            {
                cur++;
                entries[cur] = entries[i];
            }
        }

        entries.resize(cur + 1);
    }
};

inline void print(vector<double>& vec, bool bAll = false)
{
    printf("Vector - Length = %d\n", (uint32_t)vec.size());

    for (uint32_t i = 0; i < vec.size(); i++)
    {
        if (i > maxPrint && bAll == false) break;
        printf("vec[%d]=%g\n", i, vec[i]);
    }

    printf("\n");
}