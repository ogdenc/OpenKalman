/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for DiagonalAdapter
 */

#ifndef OPENKALMAN_DIAGONALADAPTER_HPP
#define OPENKALMAN_DIAGONALADAPTER_HPP

namespace OpenKalman
{

#ifdef __cpp_concepts
  template<vector<0, Applicability::permitted> NestedObject>
#else
  template<typename NestedObject>
#endif
  struct DiagonalAdapter : OpenKalman::internal::AdapterBase<DiagonalAdapter<NestedObject>, NestedObject>
  {

#ifndef __cpp_concepts
    static_assert(vector<NestedObject, 0, Applicability::permitted>);
#endif

  private:

    using Base = OpenKalman::internal::AdapterBase<DiagonalAdapter, NestedObject>;

    static constexpr auto dim = index_dimension_of_v<NestedObject, 0>;

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_constructible_from_diagonal : std::false_type {};

    template<typename T>
    struct is_constructible_from_diagonal<T, std::void_t<decltype(NestedObject {diagonal_of(std::declval<T>())})>>
      : std::true_type {};
#endif

  public:

    using Scalar = scalar_type_of_t<NestedObject>;


    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr DiagonalAdapter() requires std::default_initializable<NestedObject> and (not has_dynamic_dimensions<NestedObject>)
#else
    template<typename T = NestedObject, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedObject>), int> = 0>
    constexpr DiagonalAdapter()
#endif
      : Base {} {}


    /**
     * \brief Construct from a vector, matrix, or other tensor reflecting the diagonal.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptors_may_match_with<NestedObject> Arg> requires
      (not std::is_base_of_v<DiagonalAdapter, std::decay_t<Arg>>) and std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<vector_space_descriptors_may_match_with<Arg, NestedObject> and
      (not std::is_base_of_v<DiagonalAdapter, std::decay_t<Arg>>) and std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    constexpr explicit DiagonalAdapter(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


#ifndef __cpp_concepts
  private:

    template<typename Arg, typename = void>
    struct constants_match : std::true_type {};

    template<typename Arg>
    struct constants_match<Arg, std::enable_if_t<
      constant_coefficient<NestedObject>::value != constant_diagonal_coefficient<Arg>::value>>
        : std::false_type {};

  public:
#endif


    /// Assign from another \ref diagonal_matrix.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::is_base_of_v<DiagonalAdapter, std::decay_t<Arg>>) and
      vector_space_descriptors_may_match_with<NestedObject, decltype(diagonal_of(std::declval<Arg>()))> and
      (not constant_matrix<NestedObject> or constant_diagonal_matrix<Arg>) and
      (not requires { requires constant_coefficient<NestedObject>::value != constant_diagonal_coefficient<Arg>::value; }) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (not std::is_base_of<DiagonalAdapter, std::decay_t<Arg>>::value) and
      vector_space_descriptors_may_match_with<NestedObject, decltype(diagonal_of(std::declval<Arg>()))> and
      (not constant_matrix<NestedObject> or constant_diagonal_matrix<Arg>) and constants_match<Arg>::value and
      std::is_assignable<std::add_lvalue_reference_t<NestedObject>, decltype(diagonal_of(std::declval<Arg>()))>::value, int> = 0>
#endif
    constexpr auto& operator=(Arg&& arg)
    {
      using Arg_diag = decltype(diagonal_of(std::declval<Arg>()));

      if constexpr (not vector_space_descriptors_match_with<NestedObject, Arg_diag>)
        if (not vector_space_descriptors_match(this->nested_object(), diagonal_of(std::declval<Arg>())))
          throw std::invalid_argument {"Argument to DiagonalAdapter assignment operator has non-matching vector space descriptors."};

      if constexpr (constant_matrix<NestedObject>)
      {
        if constexpr (not values::fixed<constant_coefficient<NestedObject>> or not values::fixed<constant_diagonal_coefficient<Arg>>)
          if (values::to_number(this->nested_object()) != values::to_number(diagonal_of(std::forward<Arg>(arg))))
            throw std::invalid_argument {"Argument to constant_diagonal DiagonalAdapter assignment operator has non-matching constant value."};
      }
      else
      {
        this->nested_object() = diagonal_of(std::forward<Arg>(arg));
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedObject, 0> or index_dimension_of_v<Arg, 0> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedObject, 0> or index_dimension_of<Arg, 0>::value == dim), int> = 0>
#endif
    auto& operator+=(Arg&& arg)
    {
      if constexpr (dynamic_dimension<NestedObject, 0>)
        assert(get_vector_space_descriptor<0>(this->nested_object()) == get_vector_space_descriptor<0>(arg));

      this->nested_object() += diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedObject, 0> or index_dimension_of_v<Arg, 0> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_dimension<Arg, 0> or dynamic_dimension<NestedObject, 0> or index_dimension_of<Arg, 0>::value == dim), int> = 0>
#endif
    auto& operator-=(Arg&& arg)
    {
      if constexpr (dynamic_dimension<NestedObject, 0>)
        assert(get_vector_space_descriptor<0>(this->nested_object()) == get_vector_space_descriptor<0>(arg));

      this->nested_object() -= diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


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


#ifdef __cpp_concepts
    template<typename Arg> requires (index_dimension_of_v<Arg, 0> == dim)
#else
    template<typename Arg, std::enable_if_t<(index_dimension_of<Arg, 0>::value == dim), int> = 0>
#endif
    auto& operator*=(const DiagonalAdapter<Arg>& arg)
    {
      static_assert(index_dimension_of_v<Arg, 0> == dim);
      this->nested_object() = this->nested_object().array() * arg.nested_object().array();
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, DiagonalAdapter>
    friend decltype(auto) operator-(Arg&& arg)
    {
      return to_diagonal(-nested_object(std::forward<Arg>(arg)));
    }
#else
    decltype(auto) operator-() const&
    {
      return to_diagonal(-nested_object(*this));
    }

    decltype(auto) operator-() const&&
    {
      return to_diagonal(-nested_object(std::move(*this)));
    }
#endif


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalAdapter> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(Arg&& arg, S s)
    {
      return to_diagonal(scalar_product(nested_object(std::forward<Arg>(arg)), s));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalAdapter> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(S s, Arg&& arg)
    {
      return to_diagonal(scalar_product(nested_object(std::forward<Arg>(arg)), s));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, DiagonalAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, DiagonalAdapter> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator/(Arg&& arg, S s)
    {
      return to_diagonal(scalar_quotient(nested_object(std::forward<Arg>(arg)), s));
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  /**
   * \brief Deduce DiagonalAdapter NestedObject from its constructor argument.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  explicit DiagonalAdapter(Arg&&) -> DiagonalAdapter<Arg>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename NestedObject>
    struct indexible_object_traits<DiagonalAdapter<NestedObject>>
    {
      using scalar_type = scalar_type_of_t<NestedObject>;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N)
      {
        return OpenKalman::get_vector_space_descriptor<0>(std::forward<Arg>(arg).nested_object());
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }

      // get_constant(const Arg& arg) not defined

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        return constant_coefficient {arg.nested_object()};
      }


      template<Applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedObject, b>;


      template<Applicability b>
      static constexpr bool is_square = true;


      template<TriangleType t>
      static constexpr bool is_triangular = true;


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires OpenKalman::one_dimensional<nested_object_of_t<Arg&>> and directly_accessible<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<one_dimensional<typename nested_object_of<Arg&>::type> and
        directly_accessible<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(nested_object(arg)); }


      static constexpr Layout layout = OpenKalman::one_dimensional<NestedObject> ? layout_of_v<NestedObject> : Layout::none;

    };

  } // namespace interface

} // OpenKalman



#endif //OPENKALMAN_DIAGONALADAPTER_HPP
