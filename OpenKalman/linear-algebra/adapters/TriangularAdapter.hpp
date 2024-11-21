/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for TriangularAdapter
 */

#ifndef OPENKALMAN_TRIANGULARADAPTER_HPP
#define OPENKALMAN_TRIANGULARADAPTER_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<square_shaped<Qualification::depends_on_dynamic_shape> NestedObject, TriangleType triangle_type>
    requires (index_count_v<NestedObject> <= 2)
#else
  template<typename NestedObject, TriangleType triangle_type>
#endif
  struct TriangularAdapter : OpenKalman::internal::AdapterBase<TriangularAdapter<NestedObject, triangle_type>, NestedObject>
  {

#ifndef __cpp_concepts
    static_assert(square_shaped<NestedObject, Qualification::depends_on_dynamic_shape>);
    static_assert(index_count_v<NestedObject> <= 2);
#endif

  private:

    using Base = OpenKalman::internal::AdapterBase<TriangularAdapter, NestedObject>;

    static constexpr auto dim = dynamic_dimension<NestedObject, 0> ? index_dimension_of_v<NestedObject, 1> :
      index_dimension_of_v<NestedObject, 0>;

    template<typename Arg>
    static bool constexpr dimensions_match = dimension_size_of_index_is<Arg, 0, dim, Qualification::depends_on_dynamic_shape> and
      dimension_size_of_index_is<Arg, 1, dim, Qualification::depends_on_dynamic_shape>;

  public:

    using Scalar = scalar_type_of_t<NestedObject>;


    /// Default constructor.
#ifdef __cpp_concepts
    TriangularAdapter() requires std::default_initializable<NestedObject> and (not has_dynamic_dimensions<NestedObject>)
#else
    template<typename T = NestedObject, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedObject>), int> = 0>
    TriangularAdapter()
#endif
      : Base {} {}


    /// Construct from a triangular adapter if NestedObject is non-diagonal.
#ifdef __cpp_concepts
    template<triangular_adapter Arg> requires (not std::is_base_of_v<TriangularAdapter, std::decay_t<Arg>>) and
      triangular_matrix<Arg, triangle_type> and (not diagonal_matrix<NestedObject>) and
      dimensions_match<nested_object_of_t<Arg>> and
# if OPENKALMAN_CPP_FEATURE_CONCEPTS
      requires(Arg&& arg) { NestedObject {nested_object(std::forward<Arg>(arg))}; } //-- not accepted in GCC 10.1.0
# else
      std::constructible_from<NestedObject, decltype(nested_object(std::declval<Arg&&>()))>
# endif
#else
    template<typename Arg, std::enable_if_t<triangular_adapter<Arg> and (not std::is_base_of_v<TriangularAdapter, std::decay_t<Arg>>) and
      (triangular_matrix<Arg, triangle_type>) and (not diagonal_matrix<NestedObject>) and
      dimensions_match<typename nested_object_of<Arg>::type> and
      std::is_constructible<NestedObject, decltype(nested_object(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularAdapter(Arg&& arg) : Base {nested_object(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular or square matrix if NestedObject is non-diagonal.
#ifdef __cpp_concepts
    template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires (not triangular_matrix<Arg, triangle_type>) and
      (not diagonal_matrix<NestedObject>) and dimensions_match<Arg> and std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<square_shaped<Arg, Qualification::depends_on_dynamic_shape> and
      (not triangular_matrix<Arg, triangle_type>) and (not diagonal_matrix<NestedObject>) and
      dimensions_match<Arg> and std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    explicit TriangularAdapter(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg)) throw std::invalid_argument {
            "Argument to TriangularAdapter must be a square matrix, but the argument has dimensions " +
            std::to_string(get_index_dimension_of<0>(arg)) + "×" + std::to_string(get_index_dimension_of<1>(arg)) +
            " in " + __func__ + " at line " + std::to_string(__LINE__) + " of " + __FILE__};
          return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))
    } {}


    /// Construct from a triangular, non-adapter matrix.
#ifdef __cpp_concepts
    template<triangular_matrix<triangle_type> Arg> requires (not triangular_adapter<Arg>) and
      (not has_nested_object<Arg> or (diagonal_matrix<NestedObject> and diagonal_matrix<Arg>)) and
      dimensions_match<Arg> and std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg, triangle_type> and (not triangular_adapter<Arg>) and
      (not has_nested_object<Arg> or (diagonal_matrix<NestedObject> and diagonal_matrix<Arg>)) and
      dimensions_match<Arg> and std::is_constructible<NestedObject, Arg&&>::value, int> = 0>
#endif
    TriangularAdapter(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /// Construct from a diagonal matrix if NestedObject is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::is_base_of_v<TriangularAdapter, std::decay_t<Arg>>) and
      diagonal_matrix<NestedObject> and dimensions_match<Arg> and (not std::constructible_from<NestedObject, Arg&&>) and
      requires(Arg&& arg) { NestedObject {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularAdapter, std::decay_t<Arg>>) and
      diagonal_matrix<NestedObject> and dimensions_match<Arg> and (not std::is_constructible_v<NestedObject, Arg&&>) and
      std::is_constructible<NestedObject, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularAdapter(Arg&& arg) : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular square matrix if NestedObject is diagonal.
#ifdef __cpp_concepts
    template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires (not triangular_matrix<Arg>) and diagonal_matrix<NestedObject> and
      dimensions_match<Arg> and requires(Arg&& arg) { NestedObject {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_shaped<Arg, Qualification::depends_on_dynamic_shape> and
      (not triangular_matrix<Arg>) and diagonal_matrix<NestedObject> and
      dimensions_match<Arg> and std::is_constructible<NestedObject, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    explicit TriangularAdapter(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg)) throw std::invalid_argument {
            "Argument to TriangularAdapter must be a square matrix, but the argument has dimensions " +
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
      (triangle_type != TriangleType::diagonal) and (not diagonal_matrix<NestedObject>) and
      requires(Args...args) { NestedObject {make_dense_object_from<NestedObject>(static_cast<const Scalar>(args)...)}; }
#else
    template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type != TriangleType::diagonal) and (not diagonal_matrix<NestedObject>), int> = 0>
#endif
    TriangularAdapter(Args...args)
      : Base {make_dense_object_from<NestedObject>(static_cast<const Scalar>(args)...)} {}


    /**
     * \overload
     * \brief Construct diagonal matrix from a list of scalar coefficients defining the diagonal.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (triangle_type == TriangleType::diagonal or diagonal_matrix<NestedObject>) and
      requires(Args ... args) { NestedObject {
        to_diagonal(make_dense_object_from<NestedObject>(
          std::tuple<Dimensions<sizeof...(Args)>, Dimensions<1>>{}, static_cast<const Scalar>(args)...))}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type == TriangleType::diagonal or diagonal_matrix<NestedObject>), int> = 0>
#endif
    TriangularAdapter(Args...args) : Base {to_diagonal(make_dense_object_from<NestedObject>(
      std::tuple<Dimensions<sizeof...(Args)>, Dimensions<1>>{}, static_cast<const Scalar>(args)...))} {}


    /// Assign from another \ref triangular_matrix.
#ifdef __cpp_concepts
    template<triangular_matrix<triangle_type> Arg> requires
      (not std::is_base_of_v<TriangularAdapter, std::decay_t<Arg>>) and
      vector_space_descriptors_may_match_with<NestedObject, Arg> and
      (not value::static_scalar<constant_diagonal_coefficient<NestedObject>> or
        requires { requires constant_diagonal_coefficient<NestedObject>::value == constant_diagonal_coefficient<Arg>::value; }) and
      (not (diagonal_matrix<NestedObject> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg, triangle_type> and
      (not std::is_base_of_v<TriangularAdapter, std::decay_t<Arg>>) and vector_space_descriptors_may_match_with<NestedObject, Arg> and
      (not constant_diagonal_matrix<NestedObject> or constant_diagonal_matrix<Arg>) and
      (not (diagonal_matrix<NestedObject> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not constant_diagonal_matrix<NestedObject>)
        internal::set_triangle<triangle_type>(this->nested_object(), std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<vector_space_descriptors_may_match_with<NestedObject> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<vector_space_descriptors_may_match_with<Arg, NestedObject>, int> = 0>
#endif
    auto& operator+=(const TriangularAdapter<Arg, triangle_type>& arg)
    {
      internal::set_triangle<triangle_type>(this->nested_object(), this->nested_object() + std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<vector_space_descriptors_may_match_with<NestedObject> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<vector_space_descriptors_may_match_with<Arg, NestedObject>, int> = 0>
#endif
    auto& operator-=(const TriangularAdapter<Arg, triangle_type>& arg)
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
    template<vector_space_descriptors_may_match_with<NestedObject> Arg>
#else
    template<typename Arg, std::enable_if_t<vector_space_descriptors_may_match_with<Arg, NestedObject>, int> = 0>
#endif
    auto& operator*=(const TriangularAdapter<Arg, triangle_type>& arg)
    {
      contract_in_place(this->nested_object(), to_dense_object(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, TriangularAdapter>
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
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, TriangularAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, TriangularAdapter> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(Arg&& arg, S s)
    {
      return make_triangular_matrix<triangle_type_of_v<Arg>>(nested_object(std::forward<Arg>(arg)) * s);
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, TriangularAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, TriangularAdapter> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(S s, Arg&& arg)
    {
      return make_triangular_matrix<triangle_type_of_v<Arg>>(s * nested_object(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, TriangularAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, TriangularAdapter> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
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
  template<triangular_matrix M>
#else
  template<typename M, std::enable_if_t<triangular_matrix<M>, int> = 0>
#endif
  TriangularAdapter(M&&) -> TriangularAdapter<
    std::conditional_t<triangular_adapter<M>, nested_object_of_t<M>, M>,
    triangle_type_of_v<M>>;


#ifdef __cpp_concepts
  template<hermitian_matrix<Qualification::depends_on_dynamic_shape> M> requires (not triangular_matrix<M>)
#else
  template<typename M, std::enable_if_t<hermitian_matrix<M, Qualification::depends_on_dynamic_shape> and
    (not triangular_matrix<M>), int> = 0>
#endif
  explicit TriangularAdapter(M&&) -> TriangularAdapter<
    std::conditional_t<hermitian_adapter<M>, nested_object_of_t<M>, M>,
    hermitian_adapter<M, HermitianAdapterType::upper> ? TriangleType::upper : TriangleType::lower>;


#ifdef __cpp_concepts
  template<indexible M> requires (not triangular_matrix<M>) and
    (not hermitian_matrix<M, Qualification::depends_on_dynamic_shape>)
#else
  template<typename M, std::enable_if_t<indexible<M> and (not triangular_matrix<M>) and
      (not hermitian_matrix<M, Qualification::depends_on_dynamic_shape>), int> = 0>
#endif
  explicit TriangularAdapter(M&&) -> TriangularAdapter<M, TriangleType::lower>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename NestedObject, TriangleType triangle_type>
    struct indexible_object_traits<TriangularAdapter<NestedObject, triangle_type>>
    {
      using scalar_type = scalar_type_of_t<NestedObject>;


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
        if constexpr (triangle_type == TriangleType::diagonal and not diagonal_matrix<NestedObject>)
          return constant_coefficient{OpenKalman::nested_object(arg)};
        else
          return constant_diagonal_coefficient{OpenKalman::nested_object(arg)};
      }


      template<Qualification b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedObject, b>;


      template<Qualification b>
      static constexpr bool is_square = OpenKalman::square_shaped<NestedObject, b>;


      template<TriangleType t>
      static constexpr bool is_triangular = t == TriangleType::any or triangle_type == TriangleType::diagonal or triangle_type == t or
        triangular_matrix<NestedObject, t>;


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


      static constexpr Layout layout = OpenKalman::one_dimensional<NestedObject> ? layout_of_v<NestedObject> : Layout::none;

    };

  } // namespace interface

} // namespace OpenKalman


#endif //OPENKALMAN_TRIANGULARADAPTER_HPP

