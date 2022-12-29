/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of TypedMatrixBase.
 */

#ifndef OPENKALMAN_TYPEDMATRIXBASE_HPP
#define OPENKALMAN_TYPEDMATRIXBASE_HPP

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix, fixed_index_descriptor...TypedIndex>
  requires (not std::is_rvalue_reference_v<NestedMatrix>) and (sizeof...(TypedIndex) <= 2)
#else
  template<typename Derived, typename NestedMatrix, typename...TypedIndex>
#endif
  struct TypedMatrixBase : MatrixBase<Derived, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert((fixed_index_descriptor<TypedIndex> and ...));
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(sizeof...(TypedIndex) <= 2);
#endif

  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

  protected:

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this variable.

  public:

    // -------------- //
    //  Constructors  //
    // -------------- //

    /// Default constructor.
#ifdef __cpp_concepts
    TypedMatrixBase() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    TypedMatrixBase()
#endif
      : Base {} {}


    /// Construct from a typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (row_dimension_of_v<Arg> == row_dimension_of_v<NestedMatrix>) and
      (column_dimension_of_v<Arg> == column_dimension_of_v<NestedMatrix>) and
      (fixed_index_descriptor<TypedIndex> and ...) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (row_dimension_of<Arg>::value == row_dimension_of<NestedMatrix>::value) and
      (column_dimension_of<Arg>::value == column_dimension_of<NestedMatrix>::value) and
      (fixed_index_descriptor<TypedIndex> and ...) and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    TypedMatrixBase(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a typed_matrix_nestable and an \ref index descriptor set.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg, index_descriptor...Cs>
    requires (row_dimension_of_v<Arg> == row_dimension_of_v<NestedMatrix>) and
      (column_dimension_of_v<Arg> == column_dimension_of_v<NestedMatrix>) and
      std::constructible_from<NestedMatrix, Arg&&> and
      ((dynamic_index_descriptor<Cs> or dynamic_index_descriptor<TypedIndex> or equivalent_to<Cs, TypedIndex>) and ...)
#else
    template<typename Arg, typename...Cs, std::enable_if_t<typed_matrix_nestable<Arg> and (index_descriptor<Cs> and ...) and
      (row_dimension_of<Arg>::value == row_dimension_of<NestedMatrix>::value) and
      (column_dimension_of<Arg>::value == column_dimension_of<NestedMatrix>::value) and
      std::is_constructible_v<NestedMatrix, Arg&&> and
      ((dynamic_index_descriptor<Cs> or dynamic_index_descriptor<TypedIndex> or equivalent_to<Cs, TypedIndex>) and ...), int> = 0>
#endif
    TypedMatrixBase(Arg&& arg, const Cs&...cs) noexcept
      : Base {std::forward<Arg>(arg)}, my_dimensions {cs...} {}


    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { NestedMatrix {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, const Scalar> and ...) and
      (sizeof...(Args) > 0) and (
        (diagonal_matrix<NestedMatrix> and
          std::is_constructible_v<NestedMatrix, dense_writable_matrix_t<NestedMatrix, Dimensions<sizeof...(Args)>, Dimensions<1>>>) or
        (sizeof...(Args) == row_dimension_of<NestedMatrix>::value * column_dimension_of<NestedMatrix>::value and
          std::is_constructible_v<NestedMatrix, dense_writable_matrix_t<NestedMatrix,
            Dimensions<row_dimension_of<NestedMatrix>::value>, Dimensions<column_dimension_of<NestedMatrix>::value>>>)), int> = 0>
#endif
    TypedMatrixBase(Args ... args)
      : Base {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)} {}


    // ---------------------- //
    //  Assignment Operators  //
    // ---------------------- //

    /// Assign from a \ref typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
        return *this;
      }
    }


    /// Multiply by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix() *= s;
      return *this;
    }


    /// Divide by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix() /= s;
      return *this;
    }

  private:

    std::tuple<TypedIndex...> my_dimensions;

#ifdef __cpp_concepts
    template<typename T, std::size_t N> friend struct interface::CoordinateSystemTraits;
#else
    template<typename T, std::size_t N, typename Enable> friend struct OpenKalman::interface::CoordinateSystemTraits;
#endif

  };


} // namespace OpenKalman


#endif //OPENKALMAN_TYPEDMATRIXBASE_HPP
