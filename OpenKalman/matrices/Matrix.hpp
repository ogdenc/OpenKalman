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

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // --------------------- //
  //        Matrix         //
  // --------------------- //

#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, fixed_vector_space_descriptor ColumnCoefficients, typed_matrix_nestable NestedMatrix>
  requires (dimension_size_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (dimension_size_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_vector_space_descriptor<RowCoefficients> == dynamic_dimension<NestedMatrix, 0>) and
    (dynamic_vector_space_descriptor<ColumnCoefficients> == dynamic_dimension<NestedMatrix, 1>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct Matrix : oin::TypedMatrixBase<Matrix<RowCoefficients, ColumnCoefficients, NestedMatrix>, NestedMatrix,
    RowCoefficients, ColumnCoefficients>
  {

#ifndef __cpp_concepts
    static_assert(fixed_vector_space_descriptor<RowCoefficients>);
    static_assert(fixed_vector_space_descriptor<ColumnCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(dimension_size_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>);
    static_assert(dimension_size_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(dynamic_vector_space_descriptor<RowCoefficients> == dynamic_dimension<NestedMatrix, 0>);
    static_assert(dynamic_vector_space_descriptor<ColumnCoefficients> == dynamic_dimension<NestedMatrix, 1>);
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
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      //requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } // \todo 't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not std::is_base_of_v<Matrix, std::decay_t<Arg>> and
      not euclidean_transformed<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed".
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix,
        decltype(from_euclidean<RowCoefficients>(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept
      : Base {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (index_dimension_of_v<Arg, 0> == index_dimension_of_v<NestedMatrix, 0>) and
      (index_dimension_of_v<Arg, 1> == index_dimension_of_v<NestedMatrix, 1>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (index_dimension_of<Arg, 0>::value == index_dimension_of<NestedMatrix, 0>::value) and
      (index_dimension_of<Arg, 1>::value == index_dimension_of<NestedMatrix, 1>::value) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit Matrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from compatible \ref OpenKalman::covariance "covariance".
#ifdef __cpp_concepts
    template<covariance Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {make_dense_writable_matrix_from(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, dense_writable_matrix_t<Arg>>, int> = 0>
#endif
    Matrix(Arg&& arg) noexcept : Base {make_dense_writable_matrix_from(std::forward<Arg>(arg))} {}


    /// Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, Matrix>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
      (not std::is_base_of_v<Matrix, std::decay_t<Arg>>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(nested_matrix(std::forward<Arg>(other)));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed" matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_of_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_of_t<Arg>>()))>,
        int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(other))));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Increment from another Matrix.
    auto& operator+=(const Matrix& other)
    {
      this->nested_matrix() += other.nested_matrix();
      return *this;
    }

    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>, int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      this->nested_matrix() += nested_matrix(std::forward<Arg>(other));
      return *this;
    }


    /// Add a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (euclidean_vector_space_descriptor<ColumnCoefficients>) and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (euclidean_vector_space_descriptor<ColumnCoefficients>) and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      apply_columnwise([&arg](auto& col) { col += arg().nested_matrix(); }, this->nested_matrix());
      return *this;
    }


    /// Decrement from another Matrix.
    auto& operator-=(const Matrix& other)
    {
      this->nested_matrix() -= other.nested_matrix();
      return *this;
    }


    /// Decrement from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>, int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      this->nested_matrix() -= nested_matrix(std::forward<Arg>(other));
      return *this;
    }


    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (euclidean_vector_space_descriptor<ColumnCoefficients>) and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (euclidean_vector_space_descriptor<ColumnCoefficients>) and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      apply_columnwise([&arg](auto& col){ col -= arg().nested_matrix(); }, this->nested_matrix());
      return *this;
    }

  private:

    template<typename CR = RowCoefficients, typename CC = ColumnCoefficients, typename Arg>
    static auto make(Arg&& arg) noexcept
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
  template<typed_matrix_nestable M, vector_space_descriptor...Cs>
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
    passable_t<nested_matrix_of_t<V>>>;


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
      nested_matrix(std::forward<V>(std::declval<V>()))))>;


  /// Deduce parameter types from a Covariance.
#ifdef __cpp_concepts
  template<covariance V>
#else
  template<typename V, std::enable_if_t<covariance<V>, int> = 0>
#endif
  Matrix(V&&) -> Matrix<
    vector_space_descriptor_of_t<V, 0>,
    vector_space_descriptor_of_t<V, 0>,
    dense_writable_matrix_t<nested_matrix_of_t<V>>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a Matrix object from a typed_matrix_nestable, specifying the row and column coefficients.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
   * \tparam M A typed_matrix_nestable with size matching RowCoefficients and ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, fixed_vector_space_descriptor ColumnCoefficients, typed_matrix_nestable M>
    requires (index_dimension_of_v<M, 0> == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of_v<M, 1> == dimension_size_of_v<ColumnCoefficients>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    fixed_vector_space_descriptor<RowCoefficients> and fixed_vector_space_descriptor<ColumnCoefficients> and typed_matrix_nestable<M> and
    (index_dimension_of<M, 0>::value == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of<M, 1>::value == dimension_size_of_v<ColumnCoefficients>), int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    return Matrix<RowCoefficients, ColumnCoefficients, passable_t<M>>(std::forward<M>(m));
  }


  /**
   * \brief Make a Matrix object from a typed_matrix_nestable, specifying only the row coefficients.
   * \details The column coefficients are default Axis.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching RowCoefficients and ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, typed_matrix_nestable M>
  requires (index_dimension_of_v<M, 0> == dimension_size_of_v<RowCoefficients>)
#else
  template<typename RowCoefficients, typename M, std::enable_if_t<
    fixed_vector_space_descriptor<RowCoefficients> and typed_matrix_nestable<M> and
    (index_dimension_of<M, 0>::value == dimension_size_of_v<RowCoefficients>), int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    using ColumnCoefficients = Dimensions<index_dimension_of_v<M, 1>>;
    return Matrix<RowCoefficients, ColumnCoefficients, passable_t<M>>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Matrix object from a typed_matrix_nestable object, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    using RowCoeffs = Dimensions<index_dimension_of_v<M, 0>>;
    using ColCoeffs = Dimensions<index_dimension_of_v<M, 1>>;
    return make_matrix<RowCoeffs, ColCoeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Matrix object from a covariance object.
   * \tparam M A covariance object (i.e., Covariance, SquareRootCovariance).
   */
#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto make_matrix(M&& arg)
  {
    using C = vector_space_descriptor_of_t<M, 0>;
    return make_matrix<C, C>(make_dense_writable_matrix_from(std::forward<M>(arg)));
  }


  /**
   * \overload
   * \brief Make a Matrix object from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto make_matrix(Arg&& arg)
  {
    using RowCoeffs = vector_space_descriptor_of_t<Arg, 0>;
    using ColCoeffs = vector_space_descriptor_of_t<Arg, 1>;
    if constexpr(euclidean_transformed<Arg>)
      return make_matrix<RowCoeffs, ColCoeffs>(nested_matrix(from_euclidean<RowCoeffs>(std::forward<Arg>(arg))));
    else
      return make_matrix<RowCoeffs, ColCoeffs>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a default, self-contained Matrix object.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, fixed_vector_space_descriptor ColumnCoefficients, typed_matrix_nestable M> requires
    (index_dimension_of_v<M, 0> == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of_v<M, 1> == dimension_size_of_v<ColumnCoefficients>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    fixed_vector_space_descriptor<RowCoefficients> and fixed_vector_space_descriptor<ColumnCoefficients> and typed_matrix_nestable<M> and
    (index_dimension_of<M, 0>::value == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of<M, 1>::value == dimension_size_of_v<ColumnCoefficients>), int> = 0>
#endif
  inline auto make_matrix()
  {
    return Matrix<RowCoefficients, ColumnCoefficients, dense_writable_matrix_t<M>>();
  }


  /**
   * \overload
   * \brief Make a self-contained Matrix object with default Axis coefficients.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_matrix()
  {
    using RowCoeffs = Dimensions<index_dimension_of_v<M, 0>>;
    using ColCoeffs = Dimensions<index_dimension_of_v<M, 1>>;
    return make_matrix<RowCoeffs, ColCoeffs, M>();
  }


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
      static constexpr auto get_index_count(const Arg& arg) { return std::tuple_size_v<decltype(arg.my_dimensions)>; }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N n)
      {
        if constexpr (static_index_value<N>)
          return std::get<N>(std::forward<Arg>(arg).my_dimensions);
        else if constexpr (equivalent_to<RowCoeffs, ColCoeffs>)
          return std::get<0>(std::forward<Arg>(arg).my_dimensions);
        else
          return std::apply(
            [](const auto&...ds, N n){ return std::array {DynamicTypedIndex<scalar_type>{ds}...}[n]; },
            arg.my_dimensions, n);
      }

      using type = std::tuple<NestedMatrix>;

      static constexpr bool has_runtime_parameters = false;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(std::forward<Arg>(arg).nested_matrix());
        return Matrix<RowCoeffs, ColCoeffs, decltype(n)> {std::move(n)};
      }

      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        return constant_coefficient{arg.nestedExpression()};
      }

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<RowCoeffs> and euclidean_vector_space_descriptor<ColCoeffs>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<NestedMatrix, b>;

      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = equivalent_to<RowCoeffs, ColCoeffs> and triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_hermitian = equivalent_to<RowCoeffs, ColCoeffs> and hermitian_matrix<NestedMatrix>;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires element_gettable<nested_matrix_of_t<Arg&&>, sizeof...(I)>
  #else
      template<typename Arg, typename...I, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg&&>::type, sizeof...(I)>, int> = 0>
  #endif
      static constexpr decltype(auto) get(Arg&& arg, I...i)
      {
        return get_element(nested_matrix(std::forward<Arg>(arg)), i...);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_settable<nested_matrix_of_t<Arg&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<element_settable<typename nested_matrix_of<Arg&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        set_element(nested_matrix(arg), s, i, is...);
      }


      static constexpr bool is_writable = library_interface<std::decay_t<NestedMatrix>>::is_writable;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires directly_accessible<nested_matrix_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<directly_accessible<typename nested_matrix_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto*
      data(Arg& arg) { return internal::raw_data(nested_matrix(arg)); }


      static constexpr Layout layout = layout_of_v<NestedMatrix>;

    };

  } // namespace interface

} // namespace OpenKalman


#endif //OPENKALMAN_MATRIX_HPP
