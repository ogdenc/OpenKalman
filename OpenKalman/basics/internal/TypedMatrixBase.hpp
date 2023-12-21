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
 * \todo Incorporate \ref dynamic_vector_space_descriptor
 */

#ifndef OPENKALMAN_TYPEDMATRIXBASE_HPP
#define OPENKALMAN_TYPEDMATRIXBASE_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Base class for means or matrices.
   * \tparam Derived The derived class (e.g., Matrix, Mean, EuclideanMean).
   * \tparam NestedMatrix The nested matrix.
   * \tparam TypedIndex The \ref OpenKalman::coefficients "coefficients" representing the rows and columns of the matrix.
   */
#ifdef __cpp_concepts
  template<indexible Derived, indexible NestedMatrix, fixed_vector_space_descriptor...TypedIndex>
  requires (not std::is_rvalue_reference_v<NestedMatrix>) and (sizeof...(TypedIndex) <= 2)
#else
  template<typename Derived, typename NestedMatrix, typename...TypedIndex>
#endif
  struct TypedMatrixBase : MatrixBase<Derived, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert((fixed_vector_space_descriptor<TypedIndex> and ...));
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
    template<typed_matrix_nestable Arg> requires (index_dimension_of_v<Arg, 0> == index_dimension_of_v<NestedMatrix, 0>) and
      (index_dimension_of_v<Arg, 1> == index_dimension_of_v<NestedMatrix, 1>) and
      (fixed_vector_space_descriptor<TypedIndex> and ...) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (index_dimension_of<Arg, 0>::value == index_dimension_of<NestedMatrix, 0>::value) and
      (index_dimension_of<Arg, 1>::value == index_dimension_of<NestedMatrix, 1>::value) and
      (fixed_vector_space_descriptor<TypedIndex> and ...) and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    TypedMatrixBase(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a typed_matrix_nestable and a \ref vector_space_descriptor object set.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg, vector_space_descriptor...Cs>
    requires (index_dimension_of_v<Arg, 0> == index_dimension_of_v<NestedMatrix, 0>) and
      (index_dimension_of_v<Arg, 1> == index_dimension_of_v<NestedMatrix, 1>) and
      std::constructible_from<NestedMatrix, Arg&&> and
      ((dynamic_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<TypedIndex> or equivalent_to<Cs, TypedIndex>) and ...)
#else
    template<typename Arg, typename...Cs, std::enable_if_t<typed_matrix_nestable<Arg> and (vector_space_descriptor<Cs> and ...) and
      (index_dimension_of<Arg, 0>::value == index_dimension_of<NestedMatrix, 0>::value) and
      (index_dimension_of<Arg, 1>::value == index_dimension_of<NestedMatrix, 1>::value) and
      std::is_constructible_v<NestedMatrix, Arg&&> and
      ((dynamic_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<TypedIndex> or equivalent_to<Cs, TypedIndex>) and ...), int> = 0>
#endif
    TypedMatrixBase(Arg&& arg, const Cs&...cs) noexcept
      : Base {std::forward<Arg>(arg)}, my_dimensions {cs...} {}


    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (not diagonal_matrix<NestedMatrix> or
        requires(Args...args) {
          NestedMatrix {make_dense_object_from<decltype(diagonal_of(std::declval<NestedMatrix>()))>(static_cast<const Scalar>(args)...)}; }) and
      (diagonal_matrix<NestedMatrix> or
        requires(Args...args) {
          NestedMatrix {make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...)}; })
#else
    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, const Scalar> and ...) and
      (sizeof...(Args) > 0) and diagonal_matrix<NestedMatrix> and
        (not diagonal_matrix<NestedMatrix> or
          std::is_constructible_v<NestedMatrix, decltype(make_dense_object_from<decltype(diagonal_of(std::declval<NestedMatrix>()))>(static_cast<const Scalar>(std::declval<Args>())...))>) and
        (diagonal_matrix<NestedMatrix> or
          std::is_constructible_v<NestedMatrix, decltype(make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(std::declval<Args>())...))>), int> = 0>
#endif
    TypedMatrixBase(Args...args)
      : Base {[](Args...args){
          if constexpr (diagonal_matrix<NestedMatrix>)
          {
            using Diag = decltype(diagonal_of(std::declval<NestedMatrix>()));
            return make_dense_object_from<Diag>(static_cast<const Scalar>(args)...);
          }
          else
          {
            return make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...);
          }
        }(args...)} {}


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
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
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
      this->nested_object() *= s;
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
      this->nested_object() /= s;
      return *this;
    }

  private:

    std::tuple<TypedIndex...> my_dimensions;

#ifdef __cpp_concepts
    template<typename T> friend struct interface::indexible_object_traits;
#else
    template<typename T, typename Enable> friend struct OpenKalman::interface::indexible_object_traits;
#endif

  };


} // namespace OpenKalman


#endif //OPENKALMAN_TYPEDMATRIXBASE_HPP
