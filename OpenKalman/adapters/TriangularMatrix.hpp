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
 * \file
 * \brief Definitions for TriangularMatrix
 */

#ifndef OPENKALMAN_TRIANGULARMATRIX_HPP
#define OPENKALMAN_TRIANGULARMATRIX_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<square_shaped<Qualification::depends_on_dynamic_shape> NestedMatrix, TriangleType triangle_type>
    requires (index_count_v<NestedMatrix> <= 2)
#else
  template<typename NestedMatrix, TriangleType triangle_type>
#endif
  struct TriangularMatrix : OpenKalman::internal::AdapterBase<TriangularMatrix<NestedMatrix, triangle_type>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(square_shaped<NestedMatrix, Qualification::depends_on_dynamic_shape>);
    static_assert(index_count_v<NestedMatrix> <= 2);
#endif

  private:

    using Base = OpenKalman::internal::AdapterBase<TriangularMatrix, NestedMatrix>;

    static constexpr auto dim = dynamic_dimension<NestedMatrix, 0> ? index_dimension_of_v<NestedMatrix, 1> :
      index_dimension_of_v<NestedMatrix, 0>;

    template<typename Arg>
    static bool constexpr dimensions_match = dimension_size_of_index_is<Arg, 0, dim, Qualification::depends_on_dynamic_shape> and
      dimension_size_of_index_is<Arg, 1, dim, Qualification::depends_on_dynamic_shape>;

  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


    /// Default constructor.
#ifdef __cpp_concepts
    TriangularMatrix() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    TriangularMatrix()
#endif
      : Base {} {}


    /// Construct from a triangular adapter if NestedMatrix is non-diagonal.
#ifdef __cpp_concepts
    template<triangular_adapter Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      triangular_matrix<Arg, triangle_type> and (not diagonal_matrix<NestedMatrix>) and
      dimensions_match<nested_object_of_t<Arg>> and
# if OPENKALMAN_CPP_FEATURE_CONCEPTS
      requires(Arg&& arg) { NestedMatrix {nested_object(std::forward<Arg>(arg))}; } //-- not accepted in GCC 10.1.0
# else
      std::constructible_from<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>
# endif
#else
    template<typename Arg, std::enable_if_t<triangular_adapter<Arg> and (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      (triangular_matrix<Arg, triangle_type>) and (not diagonal_matrix<NestedMatrix>) and
      dimensions_match<typename nested_object_of<Arg>::type> and
      std::is_constructible<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) : Base {nested_object(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular or square matrix if NestedMatrix is non-diagonal.
#ifdef __cpp_concepts
    template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires (not triangular_matrix<Arg, triangle_type>) and
      (not diagonal_matrix<NestedMatrix>) and dimensions_match<Arg> and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<square_shaped<Arg, Qualification::depends_on_dynamic_shape> and
      (not triangular_matrix<Arg, triangle_type>) and (not diagonal_matrix<NestedMatrix>) and
      dimensions_match<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg)) throw std::invalid_argument {
            "Argument to TriangularMatrix must be a square matrix, but the argument has dimensions " +
            std::to_string(get_index_dimension_of<0>(arg)) + "×" + std::to_string(get_index_dimension_of<1>(arg)) +
            " in " + __func__ + " at line " + std::to_string(__LINE__) + " of " + __FILE__};
          return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))
    } {}


    /// Construct from a triangular, non-adapter matrix.
#ifdef __cpp_concepts
    template<triangular_matrix<triangle_type> Arg> requires (not triangular_adapter<Arg>) and
      (not has_nested_object<Arg> or (diagonal_matrix<NestedMatrix> and diagonal_matrix<Arg>)) and
      dimensions_match<Arg> and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg, triangle_type> and (not triangular_adapter<Arg>) and
      (not has_nested_object<Arg> or (diagonal_matrix<NestedMatrix> and diagonal_matrix<Arg>)) and
      dimensions_match<Arg> and std::is_constructible<NestedMatrix, Arg&&>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /// Construct from a diagonal matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      diagonal_matrix<NestedMatrix> and dimensions_match<Arg> and (not std::constructible_from<NestedMatrix, Arg&&>) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      diagonal_matrix<NestedMatrix> and dimensions_match<Arg> and (not std::is_constructible_v<NestedMatrix, Arg&&>) and
      std::is_constructible<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular square matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires (not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      dimensions_match<Arg> and requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_shaped<Arg, Qualification::depends_on_dynamic_shape> and
      (not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      dimensions_match<Arg> and std::is_constructible<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg)) throw std::invalid_argument {
            "Argument to TriangularMatrix must be a square matrix, but the argument has dimensions " +
            std::to_string(get_index_dimension_of<0>(arg)) + "×" + std::to_string(get_index_dimension_of<1>(arg)) +
            " in " + __func__ + " at line " + std::to_string(__LINE__) + " of " + __FILE__};
          return diagonal_of(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg))
    } {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if this is not a diagonal matrix.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (triangle_type != TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires(Args...args) { NestedMatrix {make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...)}; }
#else
    template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type != TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>), int> = 0>
#endif
    TriangularMatrix(Args...args)
      : Base {make_dense_object_from<NestedMatrix>(static_cast<const Scalar>(args)...)} {}


    /**
     * \overload
     * \brief Construct diagonal matrix from a list of scalar coefficients defining the diagonal.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (triangle_type == TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {
        to_diagonal(make_dense_object_from<NestedMatrix>(
          std::tuple<Dimensions<sizeof...(Args)>, Dimensions<1>>{}, static_cast<const Scalar>(args)...))}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type == TriangleType::diagonal or diagonal_matrix<NestedMatrix>), int> = 0>
#endif
    TriangularMatrix(Args...args) : Base {to_diagonal(make_dense_object_from<NestedMatrix>(
      std::tuple<Dimensions<sizeof...(Args)>, Dimensions<1>>{}, static_cast<const Scalar>(args)...))} {}


    /// Assign from another \ref triangular_matrix.
#ifdef __cpp_concepts
    template<triangular_matrix<triangle_type> Arg> requires
      (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      maybe_same_shape_as<NestedMatrix, Arg> and
      (not constant_diagonal_matrix<NestedMatrix, ConstantType::static_constant> or
        requires { requires constant_diagonal_coefficient<NestedMatrix>::value == constant_diagonal_coefficient<Arg>::value; }) and
      (not (diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg, triangle_type> and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and maybe_same_shape_as<NestedMatrix, Arg> and
      (not constant_diagonal_matrix<NestedMatrix> or constant_diagonal_matrix<Arg>) and
      (not (diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not constant_diagonal_matrix<NestedMatrix>)
        internal::set_triangle<triangle_type>(this->nested_object(), std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      internal::set_triangle<triangle_type>(this->nested_object(), this->nested_object() + std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      internal::set_triangle<triangle_type>(this->nested_object(), this->nested_object() - std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      internal::set_triangle<triangle_type>(this->nested_object(), scalar_product(this->nested_object(), s));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      internal::set_triangle<triangle_type>(this->nested_object(), scalar_quotient(this->nested_object(), s));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_same_shape_as<NestedMatrix> Arg>
#else
    template<typename Arg, std::enable_if_t<maybe_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator*=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      contract_in_place(this->nested_object(), to_dense_object(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, TriangularMatrix>
    friend decltype(auto) operator-(Arg&& arg)
    {
      return make_triangular_matrix<triangle_type_of_v<Arg>>(-nested_object(std::forward<Arg>(arg)));
    }
#else
    decltype(auto) operator-() const&
    {
      return make_triangular_matrix<triangle_type>(-nested_object(*this));
    }

    decltype(auto) operator-() const&&
    {
      return make_triangular_matrix<triangle_type>(-nested_object(std::move(*this)));
    }
#endif


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, TriangularMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, TriangularMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(Arg&& arg, S s)
    {
      return make_triangular_matrix<triangle_type_of_v<Arg>>(nested_object(std::forward<Arg>(arg)) * s);
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, TriangularMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, TriangularMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(S s, Arg&& arg)
    {
      return make_triangular_matrix<triangle_type_of_v<Arg>>(s * nested_object(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, TriangularMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, TriangularMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator/(Arg&& arg, S s)
    {
      return make_triangular_matrix<triangle_type_of_v<Arg>>(nested_object(std::forward<Arg>(arg)) / s);
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<triangular_matrix<TriangleType::any> M>
#else
  template<typename M, std::enable_if_t<triangular_matrix<M, TriangleType::any>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<
    std::conditional_t<triangular_adapter<M>, passable_t<nested_object_of_t<M&&>>, passable_t<M>>,
    triangle_type_of_v<M>>;


#ifdef __cpp_concepts
  template<hermitian_matrix<Qualification::depends_on_dynamic_shape> M> requires (not triangular_matrix<M, TriangleType::any>)
#else
  template<typename M, std::enable_if_t<hermitian_matrix<M, Qualification::depends_on_dynamic_shape> and
    (not triangular_matrix<M, TriangleType::any>), int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<
    std::conditional_t<hermitian_adapter<M>, passable_t<nested_object_of_t<M&&>>, passable_t<M>>,
    hermitian_adapter<M, HermitianAdapterType::upper> ? TriangleType::upper : TriangleType::lower>;


#ifdef __cpp_concepts
  template<indexible M> requires (not triangular_matrix<M, TriangleType::any>) and
    (not hermitian_matrix<M, Qualification::depends_on_dynamic_shape>)
#else
  template<typename M, std::enable_if_t<indexible<M> and (not triangular_matrix<M, TriangleType::any>) and
      (not hermitian_matrix<M, Qualification::depends_on_dynamic_shape>), int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename NestedMatrix, TriangleType triangle_type>
    struct indexible_object_traits<TriangularMatrix<NestedMatrix, triangle_type>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N n)
      {
        return internal::best_vector_space_descriptor(
          OpenKalman::get_vector_space_descriptor<0>(nested_object(arg)),
          OpenKalman::get_vector_space_descriptor<1>(nested_object(arg)));
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (triangle_type == TriangleType::diagonal and not diagonal_matrix<NestedMatrix>)
          return constant_coefficient{OpenKalman::nested_object(arg)};
        else
          return constant_diagonal_coefficient{OpenKalman::nested_object(arg)};
      }


      template<Qualification b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedMatrix, b>;


      template<TriangleType t>
      static constexpr bool is_triangular = t == TriangleType::any or triangle_type == TriangleType::diagonal or triangle_type == t or
        triangular_matrix<NestedMatrix, t>;


      static constexpr bool is_triangular_adapter = true;


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires OpenKalman::one_dimensional<nested_object_of_t<Arg&>> and raw_data_defined_for<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<one_dimensional<typename nested_object_of<Arg&>::type> and
        raw_data_defined_for<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(OpenKalman::nested_object(arg)); }


      static constexpr Layout layout = OpenKalman::one_dimensional<NestedMatrix> ? layout_of_v<NestedMatrix> : Layout::none;

    };

  } // namespace interface

} // namespace OpenKalman


#endif //OPENKALMAN_TRIANGULARMATRIX_HPP

