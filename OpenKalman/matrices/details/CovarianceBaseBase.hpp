/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEBASEBASE_H
#define OPENKALMAN_COVARIANCEBASEBASE_H

namespace OpenKalman::internal
{
  /**
   * Ultimate base of Covariance and SquareRootCovariance classes.
   */
  template<typename Derived, typename ArgType>
  struct CovarianceBaseBase : MatrixTraits<ArgType>::template CovarianceBaseType<Derived>
  {
    using BaseMatrix = ArgType;
    using Base = typename MatrixTraits<ArgType>::template CovarianceBaseType<Derived>;

    /// Default constructor.
    CovarianceBaseBase() : m_arg() {}

    /// Copy constructor.
    CovarianceBaseBase(const CovarianceBaseBase& other) : m_arg(other.m_arg) {}

    /// Move constructor.
    CovarianceBaseBase(CovarianceBaseBase&& other) noexcept : m_arg(std::move(other.m_arg)) {}

    /// Construct from another covariance.
    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    CovarianceBaseBase(Arg&& arg) noexcept : m_arg(internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg))) {}

    /// Construct from a covariance base.
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    CovarianceBaseBase(Arg&& arg) noexcept : m_arg(std::forward<Arg>(arg)) {}

    /// Copy assignment operator.
    auto& operator=(const CovarianceBaseBase& other)
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        m_arg = other.m_arg;
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBaseBase&& other) noexcept
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        m_arg = std::move(other.m_arg);
      return *this;
    }

    /// Assign from a covariance base.
    template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>, int> = 0>
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else
      {
        m_arg = std::forward<Arg>(arg);
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


  protected:
    BaseMatrix m_arg; ///< The base matrix for Covariance or SquareRootCovariance.

    template<typename, typename>
    friend struct CovarianceBaseBase;
  };


}

#endif //OPENKALMAN_COVARIANCEBASEBASE_H