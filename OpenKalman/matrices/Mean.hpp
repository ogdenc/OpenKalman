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
 * \file
 * \brief Definitions relating to Mean.
 */
#ifndef OPENKALMAN_MEAN_HPP
#define OPENKALMAN_MEAN_HPP

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // ------------------- //
  //        Mean         //
  // ------------------- //

  /// A typed vector.
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, typed_matrix_nestable NestedMatrix> requires
    (dimension_size_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct Mean : oin::TypedMatrixBase<
    Mean<RowCoefficients, NestedMatrix>, NestedMatrix, RowCoefficients, Dimensions<index_dimension_of_v<NestedMatrix, 1>>>
  {

#ifndef __cpp_concepts
    static_assert(fixed_vector_space_descriptor<RowCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(dimension_size_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  protected:

    using ColumnCoefficients = Dimensions<index_dimension_of_v<NestedMatrix, 1>>;

  private:

    using Base = oin::TypedMatrixBase<Mean, NestedMatrix, RowCoefficients, ColumnCoefficients>;

  public:

    using Base::Base;


    /// Construct from a compatible mean.
#ifdef __cpp_concepts
    template<mean Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      //requires(Arg&& arg) { NestedMatrix {nested_object(std::forward<Arg>(arg))}; } // \todo doesn't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<mean<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {nested_object(std::forward<Arg>(arg))} {}


    /// Construct from a compatible Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix{from_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix,
        decltype(from_euclidean<RowCoefficients>(nested_object(std::declval<Arg&&>())))>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {from_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(arg)))} {}


    /// Construct from a compatible typed matrix or Euclidean-transformed mean.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and (not mean<Arg>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {wrap_angles<RowCoefficients>(nested_object(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not euclidean_transformed<Arg> and not mean<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {wrap_angles<RowCoefficients>(nested_object(std::forward<Arg>(arg)))} {}


    /// Construct from a typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (index_dimension_of_v<Arg, 0> == index_dimension_of_v<NestedMatrix, 0>) and
      (index_dimension_of_v<Arg, 1> == index_dimension_of_v<NestedMatrix, 1>) and
      requires(Arg&& arg) { NestedMatrix {wrap_angles<RowCoefficients>(std::declval<Arg>())}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (index_dimension_of<Arg, 0>::value == index_dimension_of<NestedMatrix, 0>::value) and
      (index_dimension_of<Arg, 1>::value == index_dimension_of<NestedMatrix, 1>::value) and
      std::is_constructible_v<NestedMatrix, decltype(wrap_angles<RowCoefficients>(std::declval<Arg>()))>, int> = 0>
#endif
    explicit Mean(Arg&& arg) noexcept : Base {wrap_angles<RowCoefficients>(std::forward<Arg>(arg))} {}


    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { NestedMatrix {wrap_angles<RowCoefficients>(
        make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...))}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == index_dimension_of<NestedMatrix, 0>::value) or
        (sizeof...(Args) == index_dimension_of<NestedMatrix, 0>::value * index_dimension_of<NestedMatrix, 1>::value)) and
      std::is_constructible_v<NestedMatrix, decltype(wrap_angles<RowCoefficients>(std::declval<NestedMatrix>()))>,
        int> = 0>
#endif
    Mean(Args ... args)
      : Base {wrap_angles<RowCoefficients>(make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...))} {}


    /**
     * \brief Assign from a compatible \ref OpenKalman::mean "mean".
     */
#ifdef __cpp_concepts
    template<mean Arg> requires (not std::derived_from<std::decay_t<Arg>, Mean>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      modifiable<NestedMatrix, nested_object_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<mean<Arg> and (not std::is_base_of_v<Mean, std::decay_t<Arg>>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      modifiable<NestedMatrix, nested_object_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(nested_object(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed".
     */
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_object_of_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_object_of_t<Arg>>()))>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(from_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(other))));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix" that is not
     * \ref OpenKalman::mean "mean" or \ref OpenKalman::euclidean_transformed "euclidean_transformed".
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not mean<Arg>) and (not euclidean_transformed<Arg>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and has_untyped_index<Arg, 1> and
      modifiable<NestedMatrix, decltype(wrap_angles<RowCoefficients>(std::declval<nested_object_of_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not mean<Arg>) and (not euclidean_transformed<Arg>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and has_untyped_index<Arg, 1>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(wrap_angles<RowCoefficients>(nested_object(std::forward<Arg>(other))));
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
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(wrap_angles<RowCoefficients>(std::forward<Arg>(arg)));
      }
      return *this;
    }


    /// Increment from another mean.
    auto& operator+=(const Mean& other)
    {
      if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
        this->nested_object() += other.nested_object();
      else
        this->nested_object() = wrap_angles<RowCoefficients>(this->nested_object() + other.nested_object());
      return *this;
    }


    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
        this->nested_object() += nested_object(std::forward<Arg>(other));
      else
        this->nested_object() = wrap_angles<RowCoefficients>(
          this->nested_object() + nested_object(std::forward<Arg>(other)));
      return *this;
    }


    /// Add a stochastic value to each column of the mean, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      apply_columnwise([&arg](auto& col){
        if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
          col += arg().nested_object();
        else
          col = wrap_angles<RowCoefficients>(col + arg().nested_object());
      }, this->nested_object());
      return *this;
    }


    /// Decrement from another mean and wrap result.
    auto& operator-=(const Mean& other)
    {
      if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
        this->nested_object() -= other.nested_object();
      else
        this->nested_object() = wrap_angles<RowCoefficients>(this->nested_object() - other.nested_object());
      return *this;
    }


    /// Decrement from another typed matrix and wrap result.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
        this->nested_object() -= nested_object(std::forward<Arg>(other));
      else
        this->nested_object() = wrap_angles<RowCoefficients>(
          this->nested_object() - nested_object(std::forward<Arg>(other)));
      return *this;
    }


    /// Subtract a stochastic value to each column of the mean, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      apply_columnwise([&arg](auto& col){
        if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
          col -= arg().nested_object();
        else
          col = wrap_angles<RowCoefficients>(col - arg().nested_object());
      }, this->nested_object());
      return *this;
    }


    /// Multiply by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
        this->nested_object() *= s;
      else
        this->nested_object() = wrap_angles<RowCoefficients>(this->nested_object() * s);
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
      if constexpr(euclidean_vector_space_descriptor<RowCoefficients>)
        this->nested_object() /= s;
      else
        this->nested_object() = wrap_angles<RowCoefficients>(this->nested_object() / s);
      return *this;
    }

  protected:

    template<typename C = RowCoefficients, typename Arg>
    static auto make(Arg&& arg) noexcept
    {
      return Mean<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  /// Deduce template parameters from a typed_matrix_nestable, assuming untyped \ref vector_space_descriptor.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
#endif
  explicit Mean(V&&) -> Mean<Dimensions<index_dimension_of_v<V, 0>>, passable_t<V>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  // \todo Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<vector_space_descriptor_of_t<V, 0>, std::decay_t<decltype(
    wrap_angles<vector_space_descriptor_of_t<V, 0>>(nested_object(std::forward<V>(std::declval<V>()))))>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  // \todo Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<vector_space_descriptor_of_t<V, 0>, std::decay_t<decltype(
    from_euclidean<vector_space_descriptor_of_t<V, 0>>(nested_object(std::forward<V>(std::declval<V>()))))>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a Mean from a typed_matrix_nestable, specifying the row fixed_vector_space_descriptor.
   * \tparam TypedIndex The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor TypedIndex, typed_matrix_nestable M> requires
    (dimension_size_of_v<TypedIndex> == index_dimension_of_v<M, 0>)
#else
  template<typename TypedIndex, typename M, std::enable_if_t<fixed_vector_space_descriptor<TypedIndex> and
    typed_matrix_nestable<M> and (dimension_size_of_v<TypedIndex> == index_dimension_of<M, 0>::value), int> = 0>
#endif
  inline auto make_mean(M&& m) noexcept
  {
    constexpr auto rows = index_dimension_of_v<M, 0>;
    using Coeffs = std::conditional_t<std::is_void_v<TypedIndex>, Dimensions<rows>, TypedIndex>;
    auto&& b = wrap_angles<Coeffs>(std::forward<M>(m)); using B = decltype(b);
    return Mean<Coeffs, passable_t<B>>(std::forward<B>(b));
  }


  /**
   * \overload
   * \brief Make a Mean from a typed_matrix_nestable object, with default Axis fixed_vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_mean(M&& m) noexcept
  {
    using Coeffs = Dimensions<index_dimension_of_v<M, 0>>;
    return make_mean<Coeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Mean from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and has_untyped_index<Arg, 1>, int> = 0>
#endif
  inline auto make_mean(Arg&& arg) noexcept
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    if constexpr(euclidean_transformed<Arg>)
      return make_mean<C>(nested_object(from_euclidean<C>(std::forward<Arg>(arg))));
    else
      return make_mean<C>(nested_object(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a default, self-contained Mean.
   * \tparam TypedIndex The coefficient types corresponding to the rows.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor TypedIndex, typed_matrix_nestable M> requires
    (index_dimension_of_v<M, 0> == dimension_size_of_v<TypedIndex>)
#else
  template<typename TypedIndex, typename M, std::enable_if_t<
    fixed_vector_space_descriptor<TypedIndex> and typed_matrix_nestable<M> and
    (index_dimension_of<M, 0>::value == dimension_size_of_v<TypedIndex>), int> = 0>
#endif
  inline auto make_mean()
  {
    return Mean<TypedIndex, dense_writable_matrix_t<M>>();
  }


  /**
   * \overload
   * \brief Make a self-contained Mean with default Axis fixed_vector_space_descriptor.
   * \tparam M a typed_matrix_nestable on which the new mean is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_mean()
  {
    return make_mean<Dimensions<index_dimension_of_v<M, 0>>, M>();
  }


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Coeffs, typename NestedMatrix>
    struct indexible_object_traits<Mean<Coeffs, NestedMatrix>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(nested_object(arg)); }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
      {
        if constexpr (static_index_value<N>)
        {
          if constexpr (n == 0_uz) return arg.my_dimension;
          else return OpenKalman::get_vector_space_descriptor(nested_object(arg), n);
        }
        else if constexpr (has_uniform_dimension_type<NestedMatrix> and equivalent_to<Coeffs, uniform_dimension_type_of<NestedMatrix>>)
        {
          return arg.my_dimension;
        }
        else
        {
          if (n == 0) return DynamicTypedIndex<scalar_type> {arg.my_dimension};
          else return DynamicTypedIndex<scalar_type> {OpenKalman::get_vector_space_descriptor(nested_object(arg), n)};
        }
      }

      using dependents = std::tuple<NestedMatrix>;

      static constexpr bool has_runtime_parameters = false;


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(OpenKalman::nested_object(std::forward<Arg>(arg)));
        return Mean<Coeffs, decltype(n)> {std::move(n)};
      }

      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        return constant_coefficient{arg.nestedExpression()};
      }

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<Coeffs>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }

      template<Qualification b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedMatrix, b>;

      template<TriangleType t, Qualification b>
      static constexpr bool is_triangular = euclidean_vector_space_descriptor<Coeffs> and triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_hermitian = euclidean_vector_space_descriptor<Coeffs> and hermitian_matrix<NestedMatrix>;


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
      template<typename Arg, typename I, typename...Is> requires element_settable<nested_object_of_t<Arg&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<element_settable<typename nested_object_of<Arg&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        if constexpr(wrapped_mean<Arg>)
        {
          const auto get_coeff = [&arg, is...] (const std::size_t row) {
            return get_component(OpenKalman::nested_object(arg), row, is...);
          };
          const auto set_coeff = [&arg, is...](const std::size_t row, const scalar_type_of_t<Arg> value) {
            set_component(nested_object(arg), value, row, is...);
          };
          set_wrapped_component<Coeffs>(Coeffs{}, i, s, set_coeff, get_coeff);
        }
        else
        {
          set_component(OpenKalman::nested_object(arg), s, i, is...);
        }
      }


      static constexpr bool is_writable = library_interface<std::decay_t<NestedMatrix>>::is_writable;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires directly_accessible<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<directly_accessible<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto*
      raw_data(Arg& arg) { return internal::raw_data(OpenKalman::nested_object(arg)); }


      static constexpr Layout layout = layout_of_v<NestedMatrix>;

    };

  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_MEAN_HPP
