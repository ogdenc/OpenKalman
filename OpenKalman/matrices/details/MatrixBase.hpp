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
 * \brief Definition of MatrixBase.
 */

#ifndef OPENKALMAN_MATRIXBASE_HPP
#define OPENKALMAN_MATRIXBASE_HPP


namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Derived, typename NestedMatrix>
#endif
  struct MatrixBase : MatrixTraits<NestedMatrix>::template MatrixBaseFrom<Derived>
  {

#ifndef __cpp_concepts
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif


    /**
     * \internal
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    MatrixBase() requires self_contained<NestedMatrix> and std::default_initializable<NestedMatrix>
      : m_arg {} {}
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      self_contained<T> and std::is_default_constructible_v<T>, int> = 0>
    MatrixBase() : m_arg {} {}
#endif


    /**
     * \internal
     * \brief Copy constructor.
     */
    MatrixBase(const MatrixBase& other) : m_arg {other.m_arg} {}


    /**
     * \internal
     * \brief Move constructor.
     */
    MatrixBase(MatrixBase&& other) noexcept : m_arg {std::move(other).m_arg} {}


    /**
     * \internal
     * \brief Construct from a nestable type.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, MatrixBase>) and
      has_same_matrix_shape<Arg, NestedMatrix>::value and
      (not self_adjoint_matrix<NestedMatrix> or self_adjoint_matrix<Arg>) and
      (not upper_triangular_matrix<NestedMatrix> or upper_triangular_matrix<Arg>) and
      (not lower_triangular_matrix<NestedMatrix> or lower_triangular_matrix<Arg>) and
      (not zero_matrix<NestedMatrix> or zero_matrix<Arg>) and
      (not identity_matrix<NestedMatrix> or identity_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<not std::is_base_of_v<MatrixBase, std::decay_t<Arg>>, int> = 0,
      std::enable_if_t<has_same_matrix_shape<Arg, NestedMatrix>::value and
      (not self_adjoint_matrix<NestedMatrix> or self_adjoint_matrix<Arg>) and
      (not upper_triangular_matrix<NestedMatrix> or upper_triangular_matrix<Arg>) and
      (not lower_triangular_matrix<NestedMatrix> or lower_triangular_matrix<Arg>) and
      (not zero_matrix<NestedMatrix> or zero_matrix<Arg>) and
      (not identity_matrix<NestedMatrix> or identity_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg>, int> = 0>
#endif
    explicit MatrixBase(Arg&& arg) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \internal
     * \brief Copy assignment operator.
     */
    auto& operator=(const MatrixBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        m_arg = other.m_arg;
      return *this;
    }


    /**
     * \internal
     * \brief Move assignment operator.
     */
    auto& operator=(MatrixBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        m_arg = std::move(other).m_arg;
      return *this;
    }


    /** \internal
     * \brief Assign from a nestable type.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, MatrixBase>) and
      modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<not std::is_base_of_v<MatrixBase, std::decay_t<Arg>>, int> = 0,
      std::enable_if_t<modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        m_arg = std::forward<Arg>(arg);
      }
      return *this;
    }


    /**
     * \brief Get the nested matrix.
     */
    auto& nested_matrix() & { return m_arg; }

    /// \overload
    decltype(auto) nested_matrix() &&
    {
      if constexpr (std::is_lvalue_reference_v<NestedMatrix>) return m_arg;
      else return std::move(m_arg);
    }

    /// \overload
    const auto& nested_matrix() const & { return m_arg; }

    /// \overload
    decltype(auto) nested_matrix() const &&
    {
      if constexpr (std::is_lvalue_reference_v<NestedMatrix>) return m_arg;
      else return std::move(m_arg);
    }


  private:

    NestedMatrix m_arg; //< The nested matrix.

  };

}

#endif //OPENKALMAN_BASEMATRIX_HPP
