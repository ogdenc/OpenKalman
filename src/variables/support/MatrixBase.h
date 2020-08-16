/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MATRIXBASE_H
#define OPENKALMAN_MATRIXBASE_H

namespace OpenKalman::internal
{
  template<typename Derived, typename ArgType>
  struct MatrixBase : MatrixTraits<ArgType>::template MatrixBaseType<Derived>
  {
    using BaseMatrix = ArgType;
    static constexpr auto columns = MatrixTraits<ArgType>::columns;

    /// Default constructor.
    MatrixBase() : m_arg() {}

    /// Copy constructor.
    MatrixBase(const MatrixBase& other) : m_arg(other.base_matrix()) {}

    /// Move constructor.
    MatrixBase(MatrixBase&& other) noexcept : m_arg(std::move(other).base_matrix()) {}

    /// Forwarding constructor.
    template<typename Arg>
    MatrixBase(Arg&& arg) noexcept : m_arg(std::forward<Arg>(arg)) {}

    /// Copy assignment operator.
    auto& operator=(const MatrixBase& other)
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        m_arg = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(MatrixBase&& other) noexcept
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        m_arg = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible matrix or covariance base.
    template<typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg> or is_covariance_base_v<Arg>, int> = 0>
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

  private:
    BaseMatrix m_arg; ///< Where the base matrix is stored.

  };

}

#endif //OPENKALMAN_BASEMATRIX_H
