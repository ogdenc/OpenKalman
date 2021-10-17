/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for Eigen3::internal::EigenDynamicBase
 * \details This is the base class for library-defined dynamic Eigen matrices.
 */

#ifndef OPENKALMAN_EIGEN3_EIGENDYNAMICBASE_HPP
#define OPENKALMAN_EIGEN3_EIGENDYNAMICBASE_HPP

#include <type_traits>

namespace OpenKalman::Eigen3::internal
{

  /**
   * \internal
   * \brief Base class for library-defined dynamic Eigen matrices.
   */
#ifdef __cpp_concepts
    template<typename Scalar, std::size_t rows, std::size_t columns>
#else
    template<typename Scalar, std::size_t rows, std::size_t columns, typename = void>
#endif
    struct EigenDynamicBase {};


    // ----------------------------- //


    /**
     * \overload \internal
     * \brief Specialization for dynamic rows and Dynamic columns
     */
    template<typename Scalar>
    struct EigenDynamicBase<Scalar, 0, 0>
    {
      /**
       * \internal
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param r Number of rows.
       * \param c Number of columns.
       */
      EigenDynamicBase(std::size_t r, std::size_t c) : rows_ {r}, cols_ {c} {}


      /**
       * \internal
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires (MatrixTraits<M>::rows == 0) and (MatrixTraits<M>::columns == 0)
#else
      template<typename M, std::enable_if_t<(MatrixTraits<M>::rows == 0) and (MatrixTraits<M>::columns == 0), int> = 0>
#endif
      EigenDynamicBase(M&& m) : rows_ {m.rows()}, cols_ {m.cols()} {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      Eigen::Index rows() const { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      Eigen::Index cols() const { return cols_; }

    private:

      const std::size_t rows_;
      const std::size_t cols_;

    };


    // ----------------------------- //


    /**
     * \overload \internal
     * \brief Specialization for dynamic rows and fixed columns
     */
#ifdef __cpp_concepts
    template<typename Scalar, std::size_t columns> requires (columns > 0)
    struct EigenDynamicBase<Scalar, 0, columns>
#else
    template<typename Scalar, std::size_t columns>
    struct EigenDynamicBase<Scalar, 0, columns, std::enable_if_t<(columns > 0)>>
#endif
    {
      /**
       * \internal
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param r Number of rows.
       */
      EigenDynamicBase(std::size_t r) : rows_ {r} {}


      /**
       * \internal
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires dynamic_rows<M> and (MatrixTraits<M>::columns == columns)
#else
      template<typename M, std::enable_if_t<dynamic_rows<M> and (MatrixTraits<M>::columns == columns), int> = 0>
#endif
      EigenDynamicBase(M&& m) : rows_ {m.rows()} {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      Eigen::Index rows() const { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index cols() { return columns; }

    private:

      const std::size_t rows_;

    };


    // ----------------------------- //


    /**
     * \overload \internal
     * \brief Specialization for fixed rows and dynamic columns
     */
#ifdef __cpp_concepts
    template<typename Scalar, std::size_t rows_> requires (rows_ > 0)
    struct EigenDynamicBase<Scalar, rows_, 0>
#else
    template<typename Scalar, std::size_t rows_>
      struct EigenDynamicBase<Scalar, rows_, 0, std::enable_if_t<(rows_ > 0)>>
#endif
    {
      /**
       * \internal
       * \brief Construct a ZeroMatrix with dynamic rows and dynamic columns.
       * \param c Number of columns.
       */
      EigenDynamicBase(std::size_t c) : cols_ {c} {}


      /**
       * \internal
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires (MatrixTraits<M>::rows == rows_) and dynamic_columns<M>
#else
      template<typename M, std::enable_if_t<(MatrixTraits<M>::rows == rows_) and dynamic_columns<M>, int> = 0>
#endif
      EigenDynamicBase(M&& m) : cols_ {m.cols()} {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index rows() { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      Eigen::Index cols() const { return cols_; }

    private:

      const std::size_t cols_;

    };


    // ----------------------------- //


    /**
     * \overload \internal
     * \brief Specialization for fixed rows and fixed columns
     */
#ifdef __cpp_concepts
    template<typename Scalar, std::size_t rows_, std::size_t columns> requires (rows_ > 0) and (columns > 0)
    struct EigenDynamicBase<Scalar, rows_, columns>
#else
    template<typename Scalar, std::size_t rows_, std::size_t columns>
    struct EigenDynamicBase<Scalar, rows_, columns, std::enable_if_t<(rows_ > 0) and (columns > 0)>>
#endif
    {
      /**
       * \internal
       * \internal
       * \brief Default constructor.
       */
      EigenDynamicBase() {};


      /**
       * \internal
       * \brief Construct a ZeroMatrix based on the shape of another matrix M.
       * \details This is designed to work with the ZeroMatrix deduction guide.
       * \tparam M The matrix to be used as a shape template. M must have the same shape as the ZeroMatrix.
       */
#ifdef __cpp_concepts
      template<typename M> requires (MatrixTraits<M>::rows == rows_) and (MatrixTraits<M>::columns == columns)
#else
      template<typename M, std::enable_if_t<
        (MatrixTraits<M>::rows == rows_) and (MatrixTraits<M>::columns == columns), int> = 0>
#endif
      EigenDynamicBase(M&& m) {}


      /// \internal \return The number of fixed rows. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index rows() { return rows_; }


      /// \internal \return The number of fixed columns. \note Required by Eigen::EigenBase.
      static constexpr Eigen::Index cols() { return columns; }

    };


} // OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN3_EIGENDYNAMICBASE_HPP
