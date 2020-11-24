/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MATRIXBASE_HPP
#define OPENKALMAN_MATRIXBASE_HPP

namespace OpenKalman::internal
{
  template<typename Derived, typename ArgType>
  struct MatrixBase : MatrixTraits<ArgType>::template MatrixBaseType<Derived>
  {
    using NestedMatrix = ArgType;
    static constexpr auto columns = MatrixTraits<ArgType>::columns;

    /// Default constructor.
    MatrixBase() : m_arg() {}

    /// Copy constructor.
    MatrixBase(const MatrixBase& other) : m_arg(other.nested_matrix()) {}

    /// Move constructor.
    MatrixBase(MatrixBase&& other) noexcept : m_arg(std::move(other).nested_matrix()) {}

    /// Forwarding constructor.
    template<typename Arg>
    MatrixBase(Arg&& arg) noexcept : m_arg(std::forward<Arg>(arg)) {}

    /// Copy assignment operator.
    auto& operator=(const MatrixBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        m_arg = other.nested_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(MatrixBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        m_arg = std::move(other).nested_matrix();
      return *this;
    }

    /// Assign from a compatible matrix or covariance_nestable.
#ifdef __cpp_concepts
    template<typename Arg> requires typed_matrix_nestable<Arg> or covariance_nestable<Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> or covariance_nestable<Arg>, int> = 0>
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

    /// Get the nested matrix.
    constexpr auto& nested_matrix() & { return m_arg; }

    /// Get the nested matrix.
    constexpr auto&& nested_matrix() && { return std::move(m_arg); }

    /// Get the nested matrix.
    constexpr const auto& nested_matrix() const & { return m_arg; }

    /// Get the nested matrix.
    constexpr const auto&& nested_matrix() const && { return std::move(m_arg); }

  private:
    NestedMatrix m_arg; ///< Where the nested matrix is stored.

  };

}

#endif //OPENKALMAN_BASEMATRIX_H
