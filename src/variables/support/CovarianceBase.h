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
  /**
   * Base of Covariance and SquareRootCovariance classes.
   */
  template<typename Derived, typename ArgType, typename Enable = void>
  struct CovarianceBase;


 /**
   * Base of Covariance and SquareRootCovariance classes, general case.
   * No conversion is necessary if either
   * (1) Derived is not a square root and the base is self-adjoint; or
   * (2) Derived is a square root and the base is triangular.
   */
   template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    (is_self_adjoint_v<ArgType> and not is_square_root_v<Derived>) or
    (is_triangular_v<ArgType> and is_square_root_v<Derived>)>>
  : MatrixTraits<ArgType>::template CovarianceBaseType<Derived>
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

    /// Construct from another covariance base.
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    CovarianceBase(Arg&& arg) noexcept : m_arg(std::forward<Arg>(arg)) {}

    /// Get the base matrix.
    constexpr auto& base_matrix() & { return m_arg; }

    /// Get the base matrix.
    constexpr auto&& base_matrix() && { return std::move(m_arg); }

    /// Get the base matrix.
    constexpr const auto& base_matrix() const & { return m_arg; }

    /// Get the base matrix.
    constexpr const auto&& base_matrix() const && { return std::move(m_arg); }

    constexpr void mark_changed() const {}

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() & { return m_arg; }

    /// Get the apparent base matrix.
    constexpr auto&& get_apparent_base_matrix() && { return std::move(m_arg); }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() const & { return m_arg; }

    /// Get the apparent base matrix.
    constexpr const auto&& get_apparent_base_matrix() const && { return std::move(m_arg); }

    /// Set the apparent base matrix.
    template<typename Arg>
    constexpr void set_apparent_base_matrix(Arg&& arg)
    {
      static_assert(
        (is_self_adjoint_v<Arg> and not is_square_root_v<Derived>) or
        (is_triangular_v<Arg> and is_square_root_v<Derived>));
      if constexpr(is_square_root_v<Derived>) m_arg = Cholesky_square(arg); else m_arg = Cholesky_factor(arg);
      m_arg = std::forward<Arg>(arg);
    }

  protected:
    BaseMatrix m_arg; ///< The base matrix for Covariance or SquareRootCovariance.
  };


  /**
   * Ultimate base of Covariance and SquareRootCovariance classes, if
   * (1) Derived is a square root and the base is not triangular (i.e., it is self-adjoint but not diagonal); or
   * (2) Derived is not a square root and the base is not self-adjoint (i.e., it is triangular but not diagonal).
   */
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    (not is_self_adjoint_v<ArgType> or is_square_root_v<Derived>) and
    (not is_triangular_v<ArgType> or not is_square_root_v<Derived>)>>
  : MatrixTraits<ArgType>::template CovarianceBaseType<Derived>
  {
    using BaseMatrix = ArgType;
    using ApparentBaseMatrix = std::conditional_t<is_square_root_v<Derived>,
      typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>,
      typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<>>;

    /// Default constructor.
    CovarianceBase() : synchronized(false) {}

    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : synchronized(other.synchronized),
        apparent_base(other.apparent_base_matrix()),
        m_arg(other.base_matrix()) {}

    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : synchronized(other.synchronized),
        apparent_base(other.apparent_base_matrix()),
        m_arg(std::move(other).base_matrix()) {}

    /// Construct from a covariance base matrix.
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    CovarianceBase(Arg&& arg) noexcept : synchronized(false), m_arg(std::forward<Arg>(arg)) { synchronized = false; }

    /// Get the base matrix.
    constexpr auto& base_matrix() & { return m_arg; }

    /// Get the base matrix.
    constexpr auto&& base_matrix() && { return std::move(m_arg); }

    /// Get the base matrix.
    constexpr const auto& base_matrix() const & { return m_arg; }

    /// Get the base matrix.
    constexpr const auto&& base_matrix() const && { return std::move(m_arg); }

    void mark_changed()
    {
      synchronized = false;
    }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() &
    {
      if (not synchronized) synchronize();
      return apparent_base;
    }

    /// Get the apparent base matrix.
    constexpr auto&& get_apparent_base_matrix() &&
    {
      if (not synchronized) synchronize();
      return std::move(apparent_base);
    }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() const &
    {
      return apparent_base;
    }

    /// Get the apparent base matrix.
    constexpr const auto&& get_apparent_base_matrix() const &&
    {
      return std::move(apparent_base);
    }

    /// Set the apparent base matrix.
    template<typename Arg>
    constexpr void set_apparent_base_matrix(Arg&& arg)
    {
      static_assert(
        (is_self_adjoint_v<Arg> and not is_square_root_v<Derived>) or
        (is_triangular_v<Arg> and is_square_root_v<Derived>));
      if constexpr(is_square_root_v<Derived>) m_arg = Cholesky_square(arg); else m_arg = Cholesky_factor(arg);
      apparent_base = std::forward<Arg>(arg);
      synchronized = true;
    }

    constexpr auto operator() (std::size_t i, std::size_t j)
    {
      if(not synchronized) synchronize();
      return apparent_base(i, j);
    }

    constexpr auto operator() (std::size_t i, std::size_t j) const
    {
      return apparent_base(i, j);
    }

  protected:
    bool synchronized;

    void synchronize()
    {
      if constexpr(is_square_root_v<Derived>)
        apparent_base = Cholesky_factor(m_arg);
      else
        apparent_base = Cholesky_square(m_arg);
      synchronized = true;
    }

    ApparentBaseMatrix apparent_base; ///< The apparent base matrix for Covariance or SquareRootCovariance.
    BaseMatrix m_arg; ///< The base matrix for Covariance or SquareRootCovariance.
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
