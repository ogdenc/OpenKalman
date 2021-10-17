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
 * \internal
 * \file
 * \brief Definitions for Eigen3::Eigen3Base
 * \todo Specialize for Matrix, Covariance, etc., so that they do not derive from Eigen::MatrixBase?
 */

#ifndef OPENKALMAN_EIGEN3BASE_HPP
#define OPENKALMAN_EIGEN3BASE_HPP

namespace OpenKalman::Eigen3::internal
{
  template<typename Derived>
  struct Eigen3Base : Eigen::MatrixBase<Derived>
  {
    using Base = Eigen::MatrixBase<Derived>;

    /// \internal \note Required by Eigen 3 for this to be used in an Eigen::CwiseBinaryOp.
    using Nested = Eigen3Base;

    /**
     * \internal
     * \return The number of fixed rows. (Required by Eigen::EigenBase).
     */
#ifdef __cpp_concepts
    static constexpr Eigen::Index rows() requires (not dynamic_rows<Derived>)
#else
    template<typename D = Derived, std::enable_if_t<(not dynamic_rows<D>), int> = 0>
    static constexpr Eigen::Index rows()
#endif
    {
      return MatrixTraits<Derived>::rows;
    }


    /**
     * \internal
     * \return The number of dynamic rows. (Required by Eigen::EigenBase).
     */
#ifdef __cpp_concepts
    Eigen::Index rows() const requires dynamic_rows<Derived>
#else
    template<typename D = Derived, std::enable_if_t<dynamic_rows<D>, int> = 0>
    Eigen::Index rows() const
#endif
    {
      return row_count(static_cast<const Derived&>(*this));
    }


    /**
     * \internal
     * \return The number of fixed columns. (Required by Eigen::EigenBase).
     */
#ifdef __cpp_concepts
    static constexpr Eigen::Index cols() requires (not dynamic_columns<Derived>)
#else
    template<typename D = Derived, std::enable_if_t<(not dynamic_columns<D>), int> = 0>
    static constexpr Eigen::Index cols()
#endif
    {
      return MatrixTraits<Derived>::columns;
    }


    /**
     * \internal
     * \return The number of dynamic rows. (Required by Eigen::EigenBase).
     */
#ifdef __cpp_concepts
    Eigen::Index cols() const requires dynamic_columns<Derived>
#else
    template<typename D = Derived, std::enable_if_t<dynamic_columns<D>, int> = 0>
    Eigen::Index cols() const
#endif
    {
      return column_count(static_cast<const Derived&>(*this));
    }


    /**
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (dynamic_rows<Derived> ? 1 : 0) + (dynamic_columns<Derived> ? 1 : 0))
#else
    template<typename D = Derived, typename...Args, std::enable_if_t<
      (std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (dynamic_rows<D> ? 1 : 0) + (dynamic_columns<D> ? 1 : 0)), int> = 0>
#endif
    static decltype(auto) zero(const Args...args)
    {
      return MatrixTraits<Derived>::zero(static_cast<std::size_t>(args)...);
    }


    /**
     * \brief Synonym for zero().
     * \deprecated Use zero() instead. Provided for compatibility with Eigen Zero() member.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args>
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...), int> = 0>
#endif
    [[deprecated("Use zero() instead.")]]
    static decltype(auto) Zero(const Args...args)
    {
      if constexpr (sizeof...(Args) == (dynamic_rows<Derived> ? 1 : 0) + (dynamic_columns<Derived> ? 1 : 0))
        return MatrixTraits<Derived>::zero(static_cast<std::size_t>(args)...);
      else
        return Base::Zero(static_cast<Eigen::Index>(args)...);
    }


    /**
     * \return A square identity matrix with the same number of rows.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires
    (sizeof...(Args) == (dynamic_shape<Derived> ? 1 : 0))
#else
    template<typename D = Derived, typename...Args, std::enable_if_t<
      (std::is_convertible_v<Args, Eigen::Index> and ...) and (sizeof...(Args) == (dynamic_shape<D> ? 1 : 0)), int> = 0>
#endif
    static decltype(auto) identity(const Args...args)
    {
      return MatrixTraits<Derived>::identity(static_cast<std::size_t>(args)...);
    }


    /**
     * \brief Synonym for identity().
     * \deprecated Use identity() instead. Provided for compatibility with Eigen Identity() member.
     * \return An identity matrix with the same or identified number of rows and columns.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> Arg, std::convertible_to<Eigen::Index> ... Args> requires
      (1 + sizeof...(Args) == (dynamic_rows<Derived> ? 1 : 0) + (dynamic_columns<Derived> ? 1 : 0))
#else
    template<typename D = Derived, typename Arg, typename...Args, std::enable_if_t<
      (std::is_convertible_v<Arg, Eigen::Index> and ... and std::is_convertible_v<Args, Eigen::Index>) and
      (1 + sizeof...(Args) == (dynamic_rows<D> ? 1 : 0) + (dynamic_columns<D> ? 1 : 0)), int> = 0>
#endif
    [[deprecated("Use identity() instead.")]]
    static decltype(auto) Identity(const Arg arg, const Args...args)
    {
      if (((arg == args) and ...)) return identity(arg, args...);
      else return Base::Identity(static_cast<Eigen::Index>(arg), static_cast<Eigen::Index>(args)...);
    }


    /**
     * \brief Synonym for identity().
     * \deprecated Use identity() instead. Provided for compatibility with Eigen Identity() member.
     * \return An identity matrix with the same or identified number of rows and columns.
     */
#ifdef __cpp_concepts
    [[deprecated("Use identity() instead.")]]
    static decltype(auto) Identity() requires (not dynamic_shape<Derived>)
#else
    template<typename D = Derived, std::enable_if_t<not dynamic_shape<D>, int> = 0>
    [[deprecated("Use identity() instead.")]]
    static decltype(auto) Identity()
#endif
    {
      return Base::Identity();
    }


  };

} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN3BASE_HPP
