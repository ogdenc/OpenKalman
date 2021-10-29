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


  private:

    NestedMatrix m_arg; //< The nested matrix.


  public:

    /**
     * \internal
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    MatrixBase() requires self_contained<NestedMatrix> and std::default_initializable<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      self_contained<T> and std::is_default_constructible_v<T>, int> = 0>
    MatrixBase()
#endif
      : m_arg {} {}


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
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<MatrixBase, std::decay_t<Arg>>) and
      has_same_matrix_shape<Arg, NestedMatrix>::value and
      (not self_adjoint_matrix<NestedMatrix> or self_adjoint_matrix<Arg>) and
      (not upper_triangular_matrix<NestedMatrix> or upper_triangular_matrix<Arg>) and
      (not lower_triangular_matrix<NestedMatrix> or lower_triangular_matrix<Arg>) and
      (not zero_matrix<NestedMatrix> or zero_matrix<Arg>) and
      (not identity_matrix<NestedMatrix> or identity_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit MatrixBase(Arg&& arg) noexcept
      : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \internal
     * \brief Copy assignment operator.
     */
    auto& operator=(const MatrixBase& other)
#ifdef __cpp_concepts
      requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#endif
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>, "Nested matrix cannot be modified.");
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        return operator=(other.m_arg);
      }
      return *this;
    }


    /**
     * \internal
     * \brief Move assignment operator.
     */
    auto& operator=(MatrixBase&& other) noexcept
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>, "Nested matrix cannot be modified.");
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        m_arg = std::move(other).m_arg;
      }
      return *this;
    }


    /** \internal
     * \brief Assign from a nestable type.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, MatrixBase>)
#else
    template<typename Arg, std::enable_if_t<not std::is_base_of_v<MatrixBase, std::decay_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>, "Nested matrix cannot be modified.");
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        m_arg = std::forward<Arg>(arg);
      }
      return *this;
    }


    /**
     * \brief Get the nested matrix.
     */
    decltype(auto) nested_matrix() & { return (m_arg); }


    /// \overload
    decltype(auto) nested_matrix() const & { return (m_arg); }


    /// \overload
    decltype(auto) nested_matrix() && { return (std::move(*this).m_arg); }


    /// \overload
    decltype(auto) nested_matrix() const && { return (std::move(*this).m_arg); }


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
     * \return A square identity matrix with the same number of rows.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (dynamic_shape<Derived> ? 1 : 0))
#else
    template<typename D = Derived, typename...Args, std::enable_if_t<
      (std::is_convertible_v<Args, std::size_t> and ...) and (sizeof...(Args) == (dynamic_shape<D> ? 1 : 0)), int> = 0>
#endif
    static decltype(auto) identity(const Args...args)
    {
      return MatrixTraits<Derived>::identity(static_cast<std::size_t>(args)...);
    }

  };

}

#endif //OPENKALMAN_BASEMATRIX_HPP
