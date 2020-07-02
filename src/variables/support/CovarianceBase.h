/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEBASE_H
#define OPENKALMAN_COVARIANCEBASE_H

namespace OpenKalman::internal
{
  template<typename Derived, typename ArgType>
  struct CovarianceBase : MatrixTraits<ArgType>::template CovarianceBaseType<Derived>
  {
    using BaseMatrix = ArgType;

    /// Default constructor.
    CovarianceBase() : m_arg() {}

    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : m_arg(other.base_matrix()) {}

    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : m_arg(std::move(other).base_matrix()) {}

    /// Construct from a special matrix object.
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    CovarianceBase(Arg&& arg) noexcept : m_arg(std::forward<Arg>(arg)) {}

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if (this != &other) m_arg = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if (this != &other) m_arg = std::move(other).base_matrix();
      return *this;
    }

    /// Get the base matrix.
    constexpr auto& base_matrix() & { return m_arg; }

    /// Get the base matrix.
    constexpr auto&& base_matrix() && { return std::move(m_arg); }

    /// Get the base matrix.
    constexpr const auto& base_matrix() const & { return m_arg; }

    /// Get the base matrix.
    constexpr const auto&& base_matrix() const && { return std::move(m_arg); }

  private:
    BaseMatrix m_arg; ///< Where the base matrix is stored.

  };


  /**
   * Convert covariance matrix to a covariance base of type T. If T is void, convert to a triangular matrix if
   * covariance is a square root, or otherwise convert to a self-adjoint matrix.
   * @tparam T Type to which Arg is to be converted (optional).
   * @tparam Arg Type of covariance matrix to be converted
   * @param arg Covariance matrix to be converted.
   * @return A covariance base.
   */
  template<typename T = void, typename Arg,
    std::enable_if_t<is_covariance_v<Arg> or is_typed_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  convert_base_matrix(Arg&& arg) noexcept
  {
    if constexpr(std::is_void_v<T>)
    {
      if constexpr(is_Cholesky_v<Arg> and not is_square_root_v<Arg>)
      {
        return Cholesky_square(std::forward<Arg>(arg).base_matrix());
      }
      else if constexpr(not is_Cholesky_v<Arg> and is_square_root_v<Arg> and not is_diagonal_v<Arg>)
      {
        return Cholesky_factor(std::forward<Arg>(arg).base_matrix());
      }
      else
      {
        return std::forward<Arg>(arg).base_matrix();
      }
    }
    else
    {
      using ArgBase = typename MatrixTraits<Arg>::BaseMatrix;
      if constexpr(is_self_adjoint_v<T> and not is_diagonal_v<T> and
        ((not is_diagonal_v<Arg> and is_triangular_v<ArgBase> )
          or (is_diagonal_v<Arg> and is_square_root_v<Arg>)))
      {
        return Cholesky_square(std::forward<Arg>(arg).base_matrix());
      }
      else if constexpr(is_triangular_v<T> and not is_diagonal_v<T> and
        ((not is_diagonal_v<Arg> and not is_triangular_v<ArgBase> )
          or (is_diagonal_v<Arg> and not is_square_root_v<Arg> )))
      {
        if constexpr(is_diagonal_v<Arg>)
        {
          return Cholesky_factor(std::forward<Arg>(arg).base_matrix());
        }
        else
        {
          using B = decltype(Cholesky_factor(std::forward<Arg>(arg).base_matrix()));
          if constexpr(is_upper_triangular_v<T> != is_upper_triangular_v<B>)
          {
            return Cholesky_factor(adjoint(std::forward<Arg>(arg).base_matrix()));
          }
          else
          {
            return Cholesky_factor(std::forward<Arg>(arg).base_matrix());
          }
        }
      }
      else if constexpr(is_triangular_v<T> and is_Cholesky_v<Arg>
        and is_upper_triangular_v<T> != is_upper_triangular_v<ArgBase>)
      {
        return adjoint(std::forward<Arg>(arg).base_matrix());
      }
      else
      {
        return std::forward<Arg>(arg).base_matrix();
      }
    }
  }


}

#endif //OPENKALMAN_COVARIANCEBASE_H
