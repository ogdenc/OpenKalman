/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MATRIX_HPP
#define OPENKALMAN_MATRIX_HPP

#include "basics/basics.hpp"

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // --------------------- //
  //        Matrix         //
  // --------------------- //

#ifdef __cpp_concepts
  template<fixed_pattern RowCoefficients, fixed_pattern ColumnCoefficients, typed_matrix_nestable NestedMatrix>
  requires (coordinates::dimension_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (coordinates::dimension_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_pattern<RowCoefficients> == dynamic_dimension<NestedMatrix, 0>) and
    (dynamic_pattern<ColumnCoefficients> == dynamic_dimension<NestedMatrix, 1>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct Matrix : oin::TypedMatrixBase<Matrix<RowCoefficients, ColumnCoefficients, NestedMatrix>, NestedMatrix,
    RowCoefficients, ColumnCoefficients>
  {

#ifndef __cpp_concepts
    static_assert(fixed_pattern<RowCoefficients>);
    static_assert(fixed_pattern<ColumnCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(coordinates::dimension_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>);
    static_assert(coordinates::dimension_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(dynamic_pattern<RowCoefficients> == dynamic_dimension<NestedMatrix, 0>);
    static_assert(dynamic_pattern<ColumnCoefficients> == dynamic_dimension<NestedMatrix, 1>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  private:

    using Base = oin::TypedMatrixBase<Matrix, NestedMatrix, RowCoefficients, ColumnCoefficients>;

  public:

    using Base::Base;


    /// Construct from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, Matrix>) and
      (not euclidean_transformed<Arg>) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      //requires(Arg&& arg) { NestedMatrix {nested_object(std::forward<Arg>(arg))}; } // \todo 't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not std::is_base_of_v<Matrix, std::decay_t<Arg>> and
      not euclidean_transformed<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      stdcompat::constructible_from<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>, int> = 0>
#endif
    Matrix(Arg&& arg) : Base {nested_object(std::forward<Arg>(arg))} {}


    /// Construct from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed".
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {from_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      stdcompat::constructible_from<NestedMatrix,
        decltype(from_euclidean<RowCoefficients>(nested_object(std::declval<Arg&&>())))>, int> = 0>
#endif
    Matrix(Arg&& arg)
      : Base {from_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(arg)))} {}


    /// Construct from compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (index_dimension_of_v<Arg, 0> == index_dimension_of_v<NestedMatrix, 0>) and
      (index_dimension_of_v<Arg, 1> == index_dimension_of_v<NestedMatrix, 1>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (index_dimension_of<Arg, 0>::value == index_dimension_of<NestedMatrix, 0>::value) and
      (index_dimension_of<Arg, 1>::value == index_dimension_of<NestedMatrix, 1>::value) and
      stdcompat::constructible_from<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit Matrix(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /// Construct from compatible \ref OpenKalman::covariance "covariance".
#ifdef __cpp_concepts
    template<covariance Arg> requires
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {to_dense_object(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, ColumnCoefficients> and
      stdcompat::constructible_from<NestedMatrix, dense_writable_matrix_t<Arg>>, int> = 0>
#endif
    Matrix(Arg&& arg) : Base {to_dense_object(std::forward<Arg>(arg))} {}


    /// Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, Matrix>) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
      (not std::is_base_of_v<Matrix, std::decay_t<Arg>>) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(nested_object(std::forward<Arg>(other)));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed" matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, decltype(from_euclidean<RowCoefficients>(std::declval<nested_object_of_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, decltype(from_euclidean<RowCoefficients>(std::declval<nested_object_of_t<Arg>>()))>,
      int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(from_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(other))));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, Arg&&>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Increment from another Matrix.
    auto& operator+=(const Matrix& other)
    {
      this->nested_object() += other.nested_object();
      return *this;
    }

    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      coordinates::compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>, int> = 0>
#endif
    auto& operator+=(Arg&& other)
    {
      this->nested_object() += nested_object(std::forward<Arg>(other));
      return *this;
    }


    /// Add a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (coordinates::euclidean_pattern<ColumnCoefficients>) and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (coordinates::euclidean_pattern<ColumnCoefficients>) and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      apply_columnwise([&arg](auto& col) { col += arg().nested_object(); }, this->nested_object());
      return *this;
    }


    /// Decrement from another Matrix.
    auto& operator-=(const Matrix& other)
    {
      this->nested_object() -= other.nested_object();
      return *this;
    }


    /// Decrement from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>and
      coordinates::compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>, int> = 0>
#endif
    auto& operator-=(Arg&& other)
    {
      this->nested_object() -= nested_object(std::forward<Arg>(other));
      return *this;
    }


    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (coordinates::euclidean_pattern<ColumnCoefficients>) and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (coordinates::euclidean_pattern<ColumnCoefficients>) and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      apply_columnwise([&arg](auto& col){ col -= arg().nested_object(); }, this->nested_object());
      return *this;
    }

  private:

    template<typename CR = RowCoefficients, typename CC = ColumnCoefficients, typename Arg>
    static auto make(Arg&& arg)
    {
      return Matrix<CR, CC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

  /// Deduce parameter types from a typed_matrix_nestable.
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  explicit Matrix(M&&) -> Matrix<Dimensions<index_dimension_of_v<M, 0>>, Dimensions<index_dimension_of_v<M, 1>>, passable_t<M>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, coordinates::pattern...Cs>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  explicit Matrix(M&&, const Cs&...) -> Matrix<Cs..., passable_t<M>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    vector_space_descriptor_of_t<V, 0>,
    vector_space_descriptor_of_t<V, 1>,
    passable_t<nested_object_of_t<V>>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  // \todo Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V> requires has_untyped_index<V, 1>
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V> and has_untyped_index<V, 1>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    vector_space_descriptor_of_t<V, 0>,
    vector_space_descriptor_of_t<V, 1>,
    decltype(from_euclidean<vector_space_descriptor_of_t<V, 0>>(
      nested_object(std::forward<V>(std::declval<V>()))))>;


  /// Deduce parameter types from a Covariance.
#ifdef __cpp_concepts
  template<covariance V>
#else
  template<typename V, std::enable_if_t<covariance<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    vector_space_descriptor_of_t<V, 0>,
    vector_space_descriptor_of_t<V, 0>,
    dense_writable_matrix_t<nested_object_of_t<V>>>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename RowCoeffs, typename ColCoeffs, typename NestedMatrix>
    struct indexible_object_traits<Matrix<RowCoeffs, ColCoeffs, NestedMatrix>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return collections::size_of_v<decltype(arg.my_dimensions)>; }

      template<typename Arg, typename N>
      static constexpr auto get_pattern_collection(Arg&& arg, N n)
      {
        if constexpr (values::fixed<N>)
          return std::get<N>(std::forward<Arg>(arg).my_dimensions);
        else if constexpr (compares_with<RowCoeffs, ColCoeffs>)
          return std::get<0>(std::forward<Arg>(arg).my_dimensions);
        else
          return std::apply(
            [](const auto&...ds, N n){ return std::array {DynamicDescriptor<scalar_type>{ds}...}[n]; },
            arg.my_dimensions, n);
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        return constant_coefficient{arg.nestedExpression()};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (coordinates::euclidean_pattern<RowCoeffs> and coordinates::euclidean_pattern<ColCoeffs>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedMatrix, b>;


      template<applicability b>
      static constexpr bool is_square = OpenKalman::square_shaped<NestedMatrix, b>;


      template<triangle_type t>
      static constexpr bool is_triangular = compares_with<RowCoeffs, ColCoeffs>and triangular_matrix<NestedMatrix, t>;


      static constexpr bool is_triangular_adapter = false;


      static constexpr bool is_hermitian = compares_with<RowCoeffs, ColCoeffs>and hermitian_matrix<NestedMatrix>;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires element_gettable<nested_object_of_t<Arg&&>, sizeof...(I)>
  #else
      template<typename Arg, typename...I, std::enable_if_t<element_gettable<typename nested_object_of<Arg&&>::type, sizeof...(I)>, int> = 0>
  #endif
      static constexpr decltype(auto) get(Arg&& arg, I...i)
      {
        return get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), i...);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires writable_by_component<nested_object_of_t<Arg&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<writable_by_component<typename nested_object_of<Arg&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        set_component(OpenKalman::nested_object(arg), s, i, is...);
      }


      static constexpr bool is_writable = library_interface<std::decay_t<NestedMatrix>>::is_writable;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires raw_data_defined_for<NestedMatrix>
#else
      template<typename Arg, std::enable_if_t<raw_data_defined_for<NestedMatrix>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(nested_object(arg)); }


      static constexpr data_layout layout = layout_of_v<NestedMatrix>;

    };

  }

}


#endif
