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

    /**
     * CSCMat::CSCMat - Compressed sparse matrix default contructor.
     */
    
    CSCMat() : n(0), m(0), name("") {}
    
    /**
     * CSCMat::CSCMat - Compressed sparse matrix constructor.
     *
     * @param rows Rows in matrix.
     * @param cols Columns in matrix.
     * @param name Name of matrix, used when printing.
     */
    
    CSCMat(uint32_t rows, uint32_t cols, string name) : n(rows), m(cols), name(name) {}

    /**
     * CSCMat::print Print CSC matrix's entries.
     *
     * @param bAll Whether to print full matrix.
     */
    
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

    /**
     * CSCMat::operator+= - Add two CSC matrices together. Add both matrices' entries to
     * a triplet matrix. Sort and merge the triplet matrix, then convert back to
     * CSC matrix.
     *
     * @param rhs CSC matrix to add to this one. 
     */
    
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

    /**
     * CSCMat::operator* - Do CSC matrix * vector multiply.
     *
     * @param rhs Vector to multiply by this matrix.
     *
     * @return Result of matrix vector multiply.
     */
    
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

    /**
     * CSCMat::toTriplet - Convert CSC matrix to triplet form.
     */
    
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

    /**
     * TripletMat::TripletMat - Triplet matrix default constructor.
     */
    
    TripletMat() : n(0), m(0), name("") {}
    
    /**
     * TripletMat::TripletMat - Triplet matrix constructor.
     *
     * @param rows Number of matrix rows.
     * @param cols Number of matrix columns.
     * @param name Name of matrix used when printing.
     */
    
    TripletMat(uint32_t rows, uint32_t cols, string name) : n(rows), m(cols), name(name) {}

    /**
     * NN::comp Comparison function used to sort triplet matrix entries. Sort in row-major
     * order.
     *
     * @param  a First triplet to compare.
     * @param  b Secodn triplet to compare.
     * @return   True if a comes before b.
     */
    
    static bool comp(const Triplet<T>& a, const Triplet<T>& b)
    {
        if (a.r < b.r) return true;
        if (a.r == b.r && a.c < b.c) return true;
        return false;
    }

    /**
     * TripletMat::insert - Add an entry to the triplet matrix.
     * 
     * @param triplet Triplet to add to matrix.
     */
    
    void insert(Triplet<T> triplet) 
    { 
        assert(triplet.r + 1 <= n);
        assert(triplet.c + 1 <= m);
        entries.push_back(triplet);
    }

    /**
     * TripletMat::print - Print a triplet matrix.
     *
     * @param bAll Whether to print full matrix.
     */
    
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

    /**
     * TripletMat::sortAndCombine - Sort a triplet matrix into row-major order and
     * combine duplicate entries (values of duplicates are summed).
     */
    
    void sortAndCombine()
    {
        sort(entries.begin(), entries.end(), comp);
        combineDuplicates();
    }

    /**
     * TripletMat::toCSC - Convert triplet matrix to compressed sparse form.
     *
     * @return - Compressed sparse representation of this matrix.
     */
    
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

    /**
     * TripletMat::combineDuplicates - Merge duplicate row/column entries. Values of duplicates
     * are summed in result.
     */
    
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

/**
 * print Print a double vector.
 *
 * @param vec  Vector to print.
 * @param bAll Whether to print full vector.
 */

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