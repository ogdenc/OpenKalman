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
    using NestedMatrix = ArgType;
    using Base = typename MatrixTraits<ArgType>::template CovarianceBaseType<Derived>;

    /// Default constructor.
    CovarianceBaseBase() : m_arg() {}

    /// Copy constructor.
    CovarianceBaseBase(const CovarianceBaseBase& other) : m_arg(other.m_arg) {}

    /// Move constructor.
    CovarianceBaseBase(CovarianceBaseBase&& other) noexcept : m_arg(std::move(other.m_arg)) {}


    /// Construct from another covariance.
#ifdef __cpp_concepts
    template<covariance Arg>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
    CovarianceBaseBase(Arg&& arg) noexcept : m_arg(internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg))) {}


    /// Construct from a covariance_nestable.
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    CovarianceBaseBase(Arg&& arg) noexcept : m_arg(std::forward<Arg>(arg)) {}


    /// Copy assignment operator.
    auto& operator=(const CovarianceBaseBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        m_arg = other.m_arg;
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBaseBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        m_arg = std::move(other.m_arg);
      return *this;
    }


    /// Assign from a covariance_nestable.
#ifdef __cpp_concepts
    template<typename Arg> requires covariance_nestable<Arg> or typed_matrix_nestable<Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> or typed_matrix_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        m_arg = std::forward<Arg>(arg);
      }
      return *this;
    }

    /**
     * \brief Get the nested matrix of this covariance matrix.
     * \details The nested matrix will be self-adjoint, triangular, or diagonal.
     * \return An lvalue reference to the nested matrix.
     */
    constexpr auto& nested_matrix() & { return m_arg; }

    /**
     * Get the nested matrix of this covariance matrix temporary.
     * \sa constexpr auto& nested_matrix() &
     * \return An rvalue reference to the nested matrix.
     */
    constexpr auto&& nested_matrix() && { return std::move(m_arg); }

    /**
     * Get the nested matrix of this constant covariance matrix.
     * \sa constexpr auto& nested_matrix() &
     * \return A constant lvalue reference to the nested matrix.
     */
    constexpr const auto& nested_matrix() const & { return m_arg; }

    /**
     * Get the nested matrix of this constant covariance matrix temporary.
     * \sa constexpr auto& nested_matrix() &
     * \return A constant rvalue reference to the nested matrix.
     */
    constexpr const auto&& nested_matrix() const && { return std::move(m_arg); }


  private:
    NestedMatrix m_arg; //< The nested matrix for Covariance or SquareRootCovariance.

    template<typename, typename>
    friend struct CovarianceBaseBase;
  };


}

#endif //OPENKALMAN_COVARIANCEBASEBASE_H
