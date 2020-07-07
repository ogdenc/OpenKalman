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


  // ============================================================================
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

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if (this != &other)
      {
        m_arg = other.m_arg;
      }
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if (this != &other)
      {
        m_arg = std::move(other.m_arg);
      }
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

    constexpr void mark_changed() const {}

    /// Get the apparent base matrix.
    constexpr auto& get_apparent_base_matrix() & { return m_arg; }

    /// Get the apparent base matrix.
    constexpr auto&& get_apparent_base_matrix() && { return std::move(m_arg); }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() const & { return m_arg; }

    /// Get the apparent base matrix.
    constexpr const auto&& get_apparent_base_matrix() const && { return std::move(m_arg); }

    /// Set the apparent base matrix.
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    constexpr void set_apparent_base_matrix(Arg&& arg)
    {
      static_assert(
        (is_self_adjoint_v<Arg> and not is_square_root_v<Derived>) or
        (is_triangular_v<Arg> and is_square_root_v<Derived>));
      if constexpr(is_square_root_v<Derived>) m_arg = Cholesky_square(arg); else m_arg = Cholesky_factor(arg);
    }

    /// Set the apparent base matrix.
    template<typename Arg, std::enable_if_t<not is_covariance_base_v<Arg>, int> = 0>
    constexpr void set_apparent_base_matrix(Arg&& arg)
    {
      m_arg = BaseMatrix {std::forward<Arg>(arg)};
    }

  protected:
    BaseMatrix m_arg; ///< The base matrix for Covariance or SquareRootCovariance.
  };


  // ============================================================================
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
        apparent_base(other.apparent_base),
        m_arg(other.m_arg) {}

    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : synchronized(other.synchronized),
        apparent_base(std::move(other.apparent_base)),
        m_arg(std::move(other.m_arg)) {}

    /// Construct from a covariance base matrix.
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    CovarianceBase(Arg&& arg) noexcept : synchronized(false), m_arg(std::forward<Arg>(arg)) { synchronized = false; }

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if (this != &other)
      {
        synchronized = other.synchronized;
        apparent_base = other.apparent_base;
        m_arg = other.m_arg;
      }
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if (this != &other)
      {
        synchronized = std::move(other.synchronized);
        apparent_base = std::move(other.apparent_base);
        m_arg = std::move(other.m_arg);
      }
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
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    constexpr void set_apparent_base_matrix(Arg&& arg)
    {
      static_assert(
        (is_self_adjoint_v<Arg> and not is_square_root_v<Derived>) or
        (is_triangular_v<Arg> and is_square_root_v<Derived>));
      apparent_base = std::forward<Arg>(arg);
      if constexpr(is_square_root_v<Derived>)
        m_arg = Cholesky_square(apparent_base);
      else
        m_arg = Cholesky_factor(apparent_base);
      synchronized = true;
    }

    /// Set the apparent base matrix.
    template<typename Arg, std::enable_if_t<not is_covariance_base_v<Arg>, int> = 0>
    constexpr void set_apparent_base_matrix(Arg&& arg)
    {
      set_apparent_base_matrix(ApparentBaseMatrix {std::forward<Arg>(arg)});
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

}

#endif //OPENKALMAN_COVARIANCEBASE_H
